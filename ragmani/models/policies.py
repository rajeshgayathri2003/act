"""Policies that map an image embedding and robot state to an action chunk.

Models only compute: losses, schedules, encoders and retrieval live in train/ and data/ and are passed in.
"""
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from detr.models.transformer import RAGActionTransformer, Transformer
from ragmani.data.renames import rename
from ragmani.data.retriever import Retriever


class CopyGrad(torch.autograd.Function):
  """Copies gradient from keys to query on backward."""

  @staticmethod
  def forward(ctx, query, embeddings, pull):  # query (B, queries, D), embeddings (B, queries * keys, D): the keys' image embeddings
    ctx.save_for_backward(query)
    ctx.pull = pull
    return embeddings.view_as(embeddings)  # default

  @staticmethod
  def backward(ctx, grad_embeddings):  # grad_embeddings (B, queries * keys, D)
    query = ctx.saved_tensors[0]  # (B, queries, D)
    batch, rows, dimension = grad_embeddings.shape
    # (B, queries, keys, D) -> mean over keys -> (B, queries, D) + pull * query
    return grad_embeddings.view(batch, query.shape[1], rows // query.shape[1], dimension).mean(dim=2) + ctx.pull * query, None, None


class StatDict(nn.Module):
  """Mean and std of one quantity, saved with the checkpoint and never trained; 0 and 1 leave it unchanged until training sets them."""

  def __init__(self, dimension):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dimension))
    self.register_buffer("std", torch.ones(dimension))

  def standardize(self, value): return (value - self.mean) / self.std

  def destandardize(self, value): return value * self.std + self.mean


def mlp(sizes):
  """Linear+ReLU between consecutive sizes, with no ReLU after the last Linear."""
  layers = [module for width_in, width_out in zip(sizes[:-2], sizes[1:-1]) for module in (nn.Linear(width_in, width_out), nn.ReLU())]
  return nn.Sequential(*layers, nn.Linear(sizes[-2], sizes[-1]))


class RagAct(nn.Module):
  """Retrieval-augmented ACT: orthogonal queries fetch demo rows, attention turns them into tokens, an ACT head decodes the chunk.

  Gayathri Rajesh's ActionModel (rag_model.py) restructured: the same layers and sizes, minus four it built and never used.
  """

  def __init__(self, config, image_dim):
    super().__init__()
    self.config = config
    features = config.feature_size  # image and state are each reduced to this many; a key joins the two

    self.image_encoder = mlp([image_dim, *config.image_layers, features])
    self.state_encoder = mlp([config.state_dim, *config.state_layers, features])
    self.state_action_next_value = nn.Linear(2 * config.state_dim + config.action_dim, config.token_dim)  # a row's state, first action, next state

    self.channels = nn.ModuleList(
      nn.Sequential(  # no activations, as in the original
        nn.Linear(2 * features, config.channel_hidden),  # 128 -> 64
        nn.Linear(config.channel_hidden, config.channel_hidden),  # 64 -> 64
        nn.Linear(config.channel_hidden, config.channel_hidden),  # 64 -> 64
      )
      for _ in range(config.queries)  # num of queries
    )
    self.query_proj = nn.Linear(config.channel_hidden, 2 * features)
    self.query_fetch = nn.Linear(2 * features, image_dim)  # q into the image embedding's space, to search with
    self.action_proj = nn.Linear(config.token_dim + 2 * features + config.channel_hidden, config.token_dim)  # attn_out [concat] q_stack gives one token

    transformer = Transformer(
      d_model=config.d_model,
      nhead=config.heads,
      num_encoder_layers=config.encoder_layers,
      num_decoder_layers=config.decoder_layers,
      dim_feedforward=config.feedforward,
      dropout=config.dropout,
      normalize_before=False,
      return_intermediate_dec=False,
    )
    self.head = RAGActionTransformer(
      transformer,
      rag_dim=config.token_dim,
      state_dim=config.state_dim,
      n_rag_tokens=config.queries,
      action_dim=config.action_dim,
      num_queries=config.chunk,
    )

    self.state = StatDict(config.state_dim)
    self.action = StatDict(config.action_dim)
    if config.add_image_emb_to_search:  # q_fetch starts at 0, so the search starts as the image's own
      nn.init.zeros_(self.query_fetch.weight)
      nn.init.zeros_(self.query_fetch.bias)

  @classmethod
  def load(cls, checkpoint_dir, config, image_dim):
    """A checkpoint saved by ragmani's trainer."""
    model = cls(config, image_dim)
    model.load_state_dict(rename(load_file(Path(checkpoint_dir) / "model.safetensors")))
    return model

  def forward(self, image_emb, state, retriever, exclude=None, truth=None, add_truth=None, add_image_emb=None):
    """Standardized action chunk (B, chunk, action_dim), padding logits (B, chunk) and the retrieved row indices (B, queries*keep).

    image_emb is (B, image_dim) and state (B, state_dim), both raw. exclude is (B,) episode ids the search skips. truth is each
    row's own raw (action chunk, next_state), added as one more key and value for the rows where add_truth (B,) is True.
    With add_image_emb_to_search, the search adds image_emb for the rows where add_image_emb (B,) is True; left None,
    as at eval and simulate, it adds it to every row while model.image_fade is 0 and to none once the image fades.
    """
    state = self.state.standardize(state)

    # encode the observation, then make one orthogonal query per channel
    h_obs = torch.cat([self.image_encoder(image_emb), self.state_encoder(state)], dim=-1)  # (B, 2*features)
    q_stack = self.orthogonalize(torch.stack([channel(h_obs) for channel in self.channels], dim=1))  # (B, queries, channel_hidden)
    q = self.query_proj(q_stack)  # (B, queries, 2*features)

    mem, idx, q_fetch = self.retrieve(q, image_emb, add_image_emb, retriever, exclude)
    self.diagnostics = {"model/h_obs_magnitude": h_obs.detach().norm(dim=-1).mean(), "model/q_fetch_magnitude": q_fetch.detach().norm(dim=-1).mean()}  # read by training's logging
    self.q_fetch = q_fetch  # read by training's retrieval loss
    z = self.generate_latent(state, h_obs, q_stack, q, q_fetch, mem, truth, add_truth)
    action, is_pad = self.head(z, state[:, None])
    return action, is_pad.squeeze(-1), idx

  def retrieve(self, q, image_emb, add_image_emb, retriever: Retriever, exclude):
    """The kept memory rows mem, their row indices idx (B, queries*keep), and the search vectors q_fetch (B, queries, image_dim)."""
    q_fetch = self.query_fetch(q)

    if add_image_emb is None:
      add_image_emb = torch.full((len(q),), self.config.add_image_emb_to_search and self.config.image_fade == 0, device=q.device) # fill a tensor with True or False

    # (B, queries, image_dim): q_fetch corrects the frame's own image search where add_image_emb, and is the whole search elsewhere
    q_search = image_emb[:, None].float() * add_image_emb[:, None, None] + q_fetch
    topk_idx = retriever.search(q_search, self.config.neighbours, exclude)  # (B, queries, neighbours), best first

    if self.training: topk_idx = topk_idx.gather(-1, torch.rand(topk_idx.shape, device=topk_idx.device).argsort(dim=-1))  # shuffled, so training keeps a random few, like dropout
    idx = topk_idx[..., :self.config.keep].flatten(1)  # every query's rows together
    return retriever.gather(idx), idx, q_fetch

  def generate_latent(self, state, h_obs, q_stack, q, q_fetch, mem, truth, add_truth):
    """The latent tokens z (B, queries, token_dim): each query attends over the memory rows, and over the truth row where add_truth."""
    mem_state = self.state.standardize(mem["state"])
    mem_image = CopyGrad.apply(q_fetch, mem["image_emb"], self.config.query_fetch_pull)

    k = torch.cat([self.image_encoder(mem_image), self.state_encoder(mem_state)], dim=-1)  # (B, rows, 2*features)
    sas = torch.cat([mem_state, self.action.standardize(mem["action"][:, :, 0]), self.state.standardize(mem["next_state"])], dim=-1)  # state, action, next state: each row's transition

    attn_bias = None  # every query attends over every row

    if truth is not None:
      action, next_state = truth
      truth_sas = torch.cat([state, self.action.standardize(action[:, 0]), self.state.standardize(next_state)], dim=-1)

      k = torch.cat([h_obs[:, None], k], dim=1)  # the truth row first
      sas = torch.cat([truth_sas[:, None], sas], dim=1)
      hide_truth = F.pad(~add_truth[:, None, None], (0, k.shape[1] - 1))  # (B, 1, rows): True on the truth row where add_truth is False
      attn_bias = k.new_zeros(hide_truth.shape).masked_fill(hide_truth, float("-inf"))

      with torch.no_grad():
        attn_weight = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(q.shape[-1]) + attn_bias, dim=-1)  # the weights scaled_dot_product_attention uses but does not return
        self.diagnostics["retrieval/attn_weight_on_truth"] = attn_weight[add_truth][..., 0].mean()  # nan without truth rows

    v = torch.cat([self.state_action_next_value(sas), k], dim=-1)  # (B, rows, token_dim + 2*features)
    attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)  # each query attends over every row
    return self.action_proj(torch.cat([attn_out, q_stack], dim=-1))

  @staticmethod
  def orthogonalize(vectors):
    """Gram-Schmidt across the query axis of (B, queries, D): each query loses its projection onto the ones before it."""
    basis = []
    for vector in vectors.unbind(dim=1):
      orthogonal = vector
      for previous in basis: orthogonal = orthogonal - (vector * previous).sum(-1, keepdim=True) / (previous * previous).sum(-1, keepdim=True) * previous
      basis.append(orthogonal)
    return torch.stack(basis, dim=1)
