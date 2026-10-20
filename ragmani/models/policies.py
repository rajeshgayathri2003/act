"""Policies that map an image embedding and robot state to an action chunk.

Models only compute: losses, schedules, encoders and retrieval live in train/ and data/ and are passed in.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from detr.models.transformer import RAGActionTransformer, Transformer
from ragmani.models.codebook import TwoHotHead


class CopyGrad(torch.autograd.Function):
  """Copies gradient from keys to query on backward."""

  @staticmethod
  def forward(ctx, query, embeddings):  # query (B, queries, D), embeddings (B, queries * keep, D)
    ctx.queries = query.shape[1]
    return embeddings.view_as(embeddings)

  @staticmethod
  def backward(ctx, grad_embeddings):
    batch, rows, dimension = grad_embeddings.shape
    return grad_embeddings.reshape(batch, ctx.queries, rows // ctx.queries, dimension).sum(dim=2), None


class StatDict(nn.Module):
  """Mean and std of one quantity, saved with the checkpoint and never trained; 0 and 1 leave it unchanged until training sets them."""

  def __init__(self, dimension):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dimension))
    self.register_buffer("std", torch.ones(dimension))


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
    self.transition_value = nn.Linear(2 * config.state_dim + config.action_dim, config.token_dim)  # a row's state, first action, next state

    self.channels = nn.ModuleList(
      nn.Sequential(  # no activations, as in the original
        nn.Linear(2 * features, config.channel_hidden),
        nn.Linear(config.channel_hidden, config.channel_hidden),
        nn.Linear(config.channel_hidden, config.channel_hidden),
      )
      for _ in range(config.queries)
    )
    self.query_fetch = nn.Linear(2 * features, image_dim)  # q into the image embedding's space, to search with
    self.query_proj = nn.Linear(config.channel_hidden, 2 * features)
    self.action_proj = nn.Linear(config.token_dim + 2 * features + config.channel_hidden, config.token_dim)  # attn_out ⊕ q_stack → one latent token

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
      codebook=self.action_codebook(config),
    )

    self.state = StatDict(config.state_dim)
    self.action = StatDict(config.action_dim)

  @staticmethod
  def action_codebook(config):
    """With the codebook head, the decoder latent straight to logits over fixed bins per action dimension, decoded to their expected value (DreamerV3)."""
    if config.action_head != "codebook": return None
    codebook = config.codebook
    return TwoHotHead(config.d_model, config.action_dim, codebook.bins, codebook.low, codebook.high, symlog=codebook.symlog)

  @property
  def codebook(self):
    """The head's TwoHotHead, or None with the regression head."""
    return self.head.codebook

  @classmethod
  def load(cls, checkpoint_dir, config, image_dim):
    """A checkpoint saved by ragmani's trainer."""
    model = cls(config, image_dim)
    model.load_state_dict(load_file(Path(checkpoint_dir) / "model.safetensors"))
    return model

  def forward(self, image, state, retriever, exclude=None, truth=None, add_truth=None):
    """Standardized action chunk (B, chunk, action_dim), padding logits (B, chunk), the retrieved row indices (B, queries*keep)
    and, with the codebook head, the bin logits (B, chunk, action_dim, bins) the chunk was decoded from (None otherwise).

    image is (B, image_dim) and state (B, state_dim), both raw. exclude is (B,) episode ids the search skips. truth is each
    row's own raw (action chunk, next_state), added as one more key and value for the rows where add_truth (B,) is True.
    """
    state = (state - self.state.mean) / self.state.std

    # encode the observation, then make one orthogonal query per channel
    h_obs = torch.cat([self.image_encoder(image), self.state_encoder(state)], dim=-1)  # (B, 2*features)
    q_stack = self.orthogonalize(torch.stack([channel(h_obs) for channel in self.channels], dim=1))  # (B, queries, channel_hidden)
    q = self.query_proj(q_stack)  # (B, queries, 2*features)

    mem, idx, q_fetch = self.retrieve(q, retriever, exclude)
    z = self.generate_latent(image, state, h_obs, q_stack, q, q_fetch, mem, truth, add_truth)
    action, is_pad, logits = self.head(z, state[:, None])
    return action, is_pad.squeeze(-1), idx, logits

  def retrieve(self, q, retriever, exclude):
    """The kept memory rows mem, their row indices idx (B, queries*keep), and the search vectors q_fetch (B, queries, image_dim)."""
    q_fetch = self.query_fetch(q)
    topk_idx = retriever.search(q_fetch, self.config.neighbours, exclude)  # (B, queries, neighbours), best first

    if self.training: topk_idx = topk_idx.gather(-1, torch.rand(topk_idx.shape, device=topk_idx.device).argsort(dim=-1))  # shuffled, so training keeps a random few, like dropout
    idx = topk_idx[..., :self.config.keep].flatten(1)  # every query's rows together
    return retriever.gather(idx), idx, q_fetch

  def generate_latent(self, image, state, h_obs, q_stack, q, q_fetch, mem, truth, add_truth):
    """The latent tokens z (B, queries, token_dim): each query attends over the memory rows, and over the truth row where add_truth."""
    mem_state = (mem["state"] - self.state.mean) / self.state.std
    mem_image = CopyGrad.apply(q_fetch, mem["image_emb"])

    k = torch.cat([self.image_encoder(mem_image), self.state_encoder(mem_state)], dim=-1)  # (B, rows, 2*features)
    sas = torch.cat([  # state, action, next state: each row's transition
      mem_state,
      (mem["action"][:, :, 0] - self.action.mean) / self.action.std,
      (mem["next_state"] - self.state.mean) / self.state.std,
    ], dim=-1)
    e = mem["image_emb"]  # the embeddings the search scored

    if truth is not None:
      action, next_state = truth
      truth_sas = torch.cat([
        state,
        (action[:, 0] - self.action.mean) / self.action.std,
        (next_state - self.state.mean) / self.state.std,
      ], dim=-1)

      k = torch.cat([h_obs[:, None], k], dim=1)  # the truth row first
      e = torch.cat([image[:, None], e], dim=1)
      sas = torch.cat([truth_sas[:, None], sas], dim=1)

    if self.config.search_scores_in_attention: attn_bias = q_fetch @ e.transpose(1, 2)  # (B, queries, rows): each row's search similarity
    else: attn_bias = k.new_zeros(k.shape[0], 1, k.shape[1])  # (B, 1, rows)

    if truth is not None:
      hide_truth = F.pad(~add_truth[:, None, None], (0, k.shape[1] - 1))  # True on the truth row where add_truth is False
      attn_bias = attn_bias.masked_fill(hide_truth, float("-inf"))

    v = torch.cat([self.transition_value(sas), k], dim=-1)  # (B, rows, token_dim + 2*features)
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
