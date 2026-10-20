"""Policies that map an image embedding and robot state to an action chunk.

Models only compute: losses, schedules, encoders and retrieval live in train/ and data/ and are passed in.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from torch.nn import Transformer

from ragmani.config import ModelConfig
from ragmani.data.retriever import Retriever


class ImageMLP(nn.Module):
  def __init__(self, c: ModelConfig):
      super().__init__()
      layer_sizes = [c.d_image, *c.image_layers, c.d_image_out]
      self.layers = nn.Sequential()
      for dims in zip(layer_sizes, layer_sizes[1:]):
                  if self.layers: self.layers.append(c.activation())  # skips first
                  self.layers.append(nn.Linear(*dims))

  def forward(self, x): 
      return self.layers(x)

class StateMLP(nn.Module):
  def __init__(self, c: ModelConfig):
      super().__init__()
      layer_sizes = [c.d_state, *c.state_layers, c.d_state_out]
      self.layers = nn.Sequential()
      for dims in zip(layer_sizes, layer_sizes[1:]):
          if self.layers: self.layers.append(c.activation())  # skips first
          self.layers.append(nn.Linear(*dims))

  def forward(self, x): 
      return self.layers(x)

class QueryMLP(nn.Module):
  def __init__(self, c: ModelConfig):
    super().__init__()
    self.channels = nn.ModuleList(
      nn.Sequential(
          nn.Linear(c.d_key, c.d_hidden),  # d_key = d_state_out + d_image_out
          c.activation(),
          nn.Linear(c.d_hidden, c.d_hidden)
      )
      for _ in range(c.n_queries)
    )

  def forward(self, x):  # (B, combined) -> (B, n_queries, d_hidden)
    stack = torch.stack([channel(x) for channel in self.channels], dim=1)
    return orthogonalize(stack)

def orthogonalize(vectors):
  """Gram-Schmidt across the query axis of (B, queries, D): each query loses its projection onto the ones before it."""
  basis = []
  for vector in vectors.unbind(dim=1):
    orthogonal = vector
    for previous in basis: orthogonal = orthogonal - (vector * previous).sum(-1, keepdim=True) / (previous * previous).sum(-1, keepdim=True) * previous
    basis.append(orthogonal)
  return torch.stack(basis, dim=1)


class ActionHead(nn.Module):
  def __init__(self, c: ModelConfig):
    super().__init__()
    self.token_proj = nn.Linear(c.d_hidden, c.d_model)
    self.combined_proj = nn.Linear(c.d_key, c.d_model)
    self.source_pos = nn.Embedding(c.n_queries + 1, c.d_model)  # one position per source token: the query tokens and the frame
    self.action_queries = nn.Embedding(c.chunk_len, c.d_model)  # one vector per step -> decoder maps to each action

    self.transformer = (Transformer(
        d_model=c.d_model, nhead=c.n_heads, num_encoder_layers=c.encoder_layers,
        num_decoder_layers=c.decoder_layers, dim_feedforward=c.feedforward,
        dropout=c.dropout, batch_first=True))

    self.action_out = nn.Linear(c.d_model, c.d_action)
    self.is_pad_out = nn.Linear(c.d_model, 1)

  def forward(self, tokens, combined):  # tokens: (B, n_queries, d_hidden) | combined: (B, 1, d_key)
    source = torch.cat([self.token_proj(tokens), self.combined_proj(combined)], dim=1) + self.source_pos.weight  # (B, n_queries + 1, d_model), tag each token with position
    out = self.transformer(source, self.action_queries.weight.expand(len(tokens), -1, -1))  # (B, chunk_len, d_model)
    return self.action_out(out), self.is_pad_out(out)

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

  def standardize(self, value): return (value - self.mean) / self.std

  def destandardize(self, value): return value * self.std + self.mean


class RagAct(nn.Module):
  """Retrieval-augmented ACT: the frame's queries fetch demo rows, attend over what those demos did next, and an ACT-style head decodes the chunk."""
  def __init__(self, c: ModelConfig):
    super().__init__()
    self.c = c  # not self.config, which the Trainer would read as a transformers config

    self.image_mlp = ImageMLP(c)
    self.state_mlp = StateMLP(c)
    self.query_mlp = QueryMLP(c)

    self.query_fetch = nn.Linear(c.d_hidden, c.d_image, bias=False)  # W: queries to image embedding's space
    self.query_attention = nn.MultiheadAttention(c.d_hidden, c.n_attn_heads, kdim=c.d_key, vdim=c.d_action + c.d_state, batch_first=True)
    # note for vdim I did state + action versus we had 2*state + action before

    self.head = ActionHead(c) # does attention with nn.Transformer and outputs into action space
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

    self.state = StatDict(c.d_state)
    self.action = StatDict(c.d_action)

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
  def load(cls, checkpoint_dir, config):
    """A checkpoint saved by ragmani's trainer."""
    model = cls(config)
    model.load_state_dict(load_file(Path(checkpoint_dir) / "model.safetensors"))
    return model

  def forward(self, image_emb, state, retriever: Retriever, exclude=None):
    state = self.state.standardize(state)
  def forward(self, image, state, retriever, exclude=None, truth=None, add_truth=None):
    """Standardized action chunk (B, chunk, action_dim), padding logits (B, chunk), the retrieved row indices (B, queries*keep)
    and, with the codebook head, the bin logits (B, chunk, action_dim, bins) the chunk was decoded from (None otherwise).

    image is (B, image_dim) and state (B, state_dim), both raw. exclude is (B,) episode ids the search skips. truth is each
    row's own raw (action chunk, next_state), added as one more key and value for the rows where add_truth (B,) is True.
    """
    state = (state - self.state.mean) / self.state.std

    # encode the observation, then make one orthogonal query per channel
    combined = torch.cat([self.image_mlp(image_emb), self.state_mlp(state)], dim=-1)  # (B, image_out + state_out = d_key)
    q_stack = self.query_mlp(combined) # (B, n_queries, d_key)

    # retrieve step
    q_fetch = self.query_fetch(q_stack) # W q: (B, n_queries, d_image)
    topk_idx = retriever.search(q_fetch, self.c.n_neighbours, exclude)  # (B, n_queries, n_neighbours), best first
    # if self.training: topk_idx = topk_idx.gather(-1, torch.rand(topk_idx.shape, device=topk_idx.device).argsort(dim=-1))  # shuffled, so training keeps a random few, like dropout
    idx = topk_idx[..., :self.c.n_keep].flatten(1)  # every query's rows together
    mem = retriever.gather(idx)

    # keys and values
    k = torch.cat([self.image_mlp(mem["image_emb"]),
                   self.state_mlp(self.state.standardize(mem["state"]))], dim=-1)  # (B, rows, d_key)
    v = torch.cat([self.action.standardize(mem["action"][:, :, 0]),
                   self.state.standardize(mem["next_state"])], dim=-1)  # (B, rows, d_action + d_state)

    q_attn = q_fetch @ self.query_fetch.weight  # W^T W q: (B, n_queries, d_hidden)
    tokens, attn_weight = self.query_attention(q_attn, k, v)  # (B, n_queries, d_hidden), (B, n_queries, rows)

    self.diagnostics = {
      "model/combined_magnitude": combined.detach().norm(dim=-1).mean(),
      "model/q_fetch_magnitude": q_fetch.detach().norm(dim=-1).mean(),
      "retrieval/attn_entropy": -(attn_weight * attn_weight.clamp_min(1e-9).log()).sum(-1).mean().detach(),  # ln(rows) is uniform
    }

    action, is_pad = self.head(tokens, combined.detach()[:, None])  # the frame's image and state features, detached so the encoders learn only from retrieval
    return action, is_pad.squeeze(-1), idx
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
