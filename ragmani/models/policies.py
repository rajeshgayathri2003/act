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
from ragmani.models.codebook import TwoHotHead


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

def orthogonalize(vectors, eps=1e-6):
  """Gram-Schmidt across the query axis of (B, queries, D): each query loses its projection onto the ones before it; eps keeps a collapsed query from dividing by zero."""
  basis = []
  for vector in vectors.unbind(dim=1):
    orthogonal = vector
    for previous in basis: orthogonal = orthogonal - (vector * previous).sum(-1, keepdim=True) / ((previous * previous).sum(-1, keepdim=True) + eps) * previous
    basis.append(orthogonal)
  return torch.stack(basis, dim=1)


class ActionHead(nn.Module):
  def __init__(self, c: ModelConfig):
    super().__init__()
    self.token_proj = nn.Linear(c.d_hidden, c.d_model)
    self.source_pos = nn.Embedding(c.n_queries, c.d_model)  # one position per source token: the query tokens only
    self.action_queries = nn.Embedding(c.chunk_len, c.d_model)  # one vector per step -> decoder maps to each action

    self.transformer = (Transformer(
        d_model=c.d_model, nhead=c.n_heads, num_encoder_layers=c.encoder_layers,
        num_decoder_layers=c.decoder_layers, dim_feedforward=c.feedforward,
        dropout=c.dropout, batch_first=True))

    # the codebook decodes each action dimension from logits over fixed bins (DreamerV3); otherwise a linear layer regresses it
    self.codebook = TwoHotHead(c.d_model, c.d_action, c.codebook.bins, c.codebook.low, c.codebook.high, symlog=c.codebook.symlog, scale=c.codebook.scale) if c.action_head == "codebook" else None
    self.action_out = nn.Linear(c.d_model, c.d_action) if self.codebook is None else None
    self.is_pad_out = nn.Linear(c.d_model, 1)

  def forward(self, tokens):  # tokens: (B, n_queries, d_hidden)
    """Actions (B, chunk_len, d_action), padding logits (B, chunk_len, 1), and with the codebook the bin logits (B, chunk_len, d_action, bins), else None."""
    source = self.token_proj(tokens) + self.source_pos.weight  # (B, n_queries, d_model), tag each token with position
    out = self.transformer(source, self.action_queries.weight.expand(len(tokens), -1, -1))  # (B, chunk_len, d_model)
    if self.codebook is None: return self.action_out(out), self.is_pad_out(out), None
    action, logits = self.codebook(out)
    return action, self.is_pad_out(out), logits

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

    self.state = StatDict(c.d_state)
    self.action = StatDict(c.d_action)

  @classmethod
  def load(cls, checkpoint_dir, config):
    """A checkpoint saved by ragmani's trainer."""
    model = cls(config)
    model.load_state_dict(load_file(Path(checkpoint_dir) / "model.safetensors"))
    return model

  def forward(self, image_emb, state, retriever: Retriever, exclude=None):
    state = self.state.standardize(state)

    # encode the observation, then make one orthogonal query per channel
    combined = torch.cat([self.image_mlp(image_emb), self.state_mlp(state)], dim=-1)  # (B, image_out + state_out = d_key)
    q_stack = self.query_mlp(combined) # (B, n_queries, d_hidden)

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

    action, is_pad, self.action_logits = self.head(tokens)  # the logits are None without the codebook
    return action, is_pad.squeeze(-1), idx