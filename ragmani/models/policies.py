"""Policies that map an image embedding and robot state to an action chunk.

Models only compute: losses, schedules, encoders and retrieval live in train/ and data/ and are passed in.
"""
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from detr.models.transformer import RAGActionTransformer, Transformer


DRAWS = {
  "row": lambda shape: shape,       # which neighbours to keep, drawn separately for every row and query
  "batch": lambda shape: shape[-1:],  # one draw shared by the whole batch, as rag_model.py's randperm
}

RAG_MODEL_RENAMES = [  # rag_model.py ActionModel weight names to RagAct's
  (r"^image_preprocessor_query\.encoder\.", "image_encoder."),
  (r"^state_preprocessor_query\.encoder\.", "state_encoder."),
  (r"^sas_value\.", "transition_value."),
  (r"^channels\.(\d+)\.linear_(\d)\.", lambda match: f"channels.{match[1]}.{int(match[2]) - 1}."),
  (r"^act_head\.", "head."),
]
RAG_MODEL_UNUSED = ("image_preprocessor_value.", "state_preprocessor_value.", "action_preprocessor.", "key_proj.")


class Model(nn.Module):
  """Base for every model in this file: holds its config section; nn.Module already requires forward."""

  def __init__(self, config):
    super().__init__()
    self.config = config


def mlp(sizes):
  """Linear+ReLU between consecutive sizes, with no ReLU after the last Linear."""
  layers = [module for width_in, width_out in zip(sizes[:-2], sizes[1:-1]) for module in (nn.Linear(width_in, width_out), nn.ReLU())]
  return nn.Sequential(*layers, nn.Linear(sizes[-2], sizes[-1]))


class RagAct(Model):
  """Retrieval-augmented ACT: three orthogonal queries fetch demo rows, attention turns them into tokens, an ACT head decodes the chunk.

  Gayathri Rajesh's ActionModel (rag_model.py) restructured: the same layers and sizes, minus four it built and never used.
  """

  def __init__(self, config, image_dim):
    super().__init__(config)
    compressed = config.compressed  # image and state each compress to this; a key is both together
    self.image_encoder = mlp([image_dim, *config.image_layers, compressed])
    self.state_encoder = mlp([config.state_dim, *config.state_layers, compressed])
    self.transition_value = nn.Linear(2 * config.state_dim + config.action_dim, config.token_dim)  # (state, first action, next_state) of a row
    self.channels = nn.ModuleList(nn.Sequential(nn.Linear(2 * compressed, config.channel_hidden), nn.Linear(config.channel_hidden, config.channel_hidden), nn.Linear(config.channel_hidden, config.channel_hidden)) for _ in range(config.queries))  # no activations, as in the original
    self.query_fetch = nn.Linear(2 * compressed, image_dim)  # into the index's space; only feeds the search, so it gets no gradient
    self.query_proj = nn.Linear(config.channel_hidden, 2 * compressed)
    self.action_proj = nn.Linear(config.token_dim + 2 * compressed + config.channel_hidden, config.token_dim)  # attention over retrieved transitions + its query, to one token
    transformer = Transformer(
      d_model=config.d_model, nhead=config.heads, num_encoder_layers=config.encoder_layers, num_decoder_layers=config.decoder_layers,
      dim_feedforward=config.feedforward, dropout=config.dropout, normalize_before=False, return_intermediate_dec=False,
    )
    self.head = RAGActionTransformer(transformer, rag_dim=config.token_dim, state_dim=config.state_dim, n_rag_tokens=config.queries, action_dim=config.action_dim, num_queries=config.chunk)

  @classmethod
  def load(cls, checkpoint_dir, config, image_dim):
    """A checkpoint saved by ragmani's trainer, or failing that one saved by rag_model.py."""
    model = cls(config, image_dim)
    try: model.load_state_dict(load_file(Path(checkpoint_dir) / "model.safetensors"))
    except (RuntimeError, FileNotFoundError): return cls.from_rag_model(checkpoint_dir, config)
    return model

  @classmethod
  def from_rag_model(cls, checkpoint_dir, config):
    """Load an ActionModel checkpoint saved by rag_model.py; strict, so a weight left unmapped raises."""
    directory = Path(checkpoint_dir)
    weights = load_file(directory / "model.safetensors") if (directory / "model.safetensors").exists() else torch.load(directory / "pytorch_model.bin", map_location="cpu")
    renamed = {}
    for name, tensor in weights.items():
      if name.startswith(RAG_MODEL_UNUSED): continue
      for pattern, replacement in RAG_MODEL_RENAMES: name = re.sub(pattern, replacement, name)
      renamed[name] = tensor
    model = cls(config, image_dim=json.loads((directory / "config.json").read_text())["input_size_im"])
    model.load_state_dict(renamed)
    return model

  def forward(self, image, state, retriever, exclude=None, truth=None, add_truth=None):
    """Action chunk (B, chunk, action_dim) and padding logits (B, chunk) for image (B, image_dim) and state (B, state_dim).

    exclude is (B,) episode ids the search skips. truth is each row's own (action chunk, next_state), added as one more
    key and value for the rows where add_truth (B,) is True.
    """
    own_key = torch.cat([self.image_encoder(image), self.state_encoder(state)], dim=-1)  # (B, 2*compressed)
    channels = self.orthogonalize(torch.stack([channel(own_key) for channel in self.channels], dim=1))  # (B, queries, channel_hidden)
    queries = self.query_proj(channels)

    found = retriever.search(self.query_fetch(queries), self.config.neighbours, exclude)  # (B, queries, neighbours)
    kept = torch.rand(DRAWS[self.config.draw](found.shape), device=found.device).argsort(dim=-1)[..., :self.config.keep].expand(*found.shape[:-1], self.config.keep)
    rows = retriever.gather(found.gather(-1, kept).flatten(1))  # every query's rows together: (B, queries*keep, ...)
    keys = torch.cat([self.image_encoder(rows["image_emb"]), self.state_encoder(rows["state"])], dim=-1)
    transitions = torch.cat([rows["state"], rows["action"][:, :, 0], rows["next_state"]], dim=-1)

    mask = None
    if truth is not None:
      action, next_state = truth
      keys = torch.cat([own_key[:, None], keys], dim=1)
      transitions = torch.cat([torch.cat([state, action[:, 0], next_state], dim=-1)[:, None], transitions], dim=1)
      mask = torch.ones(keys.shape[:2], dtype=torch.bool, device=keys.device)
      mask[:, 0] = add_truth
      mask = mask[:, None]  # (B, 1, keys) broadcast over queries

    values = torch.cat([self.transition_value(transitions), keys], dim=-1)
    attended = F.scaled_dot_product_attention(queries, keys, values, attn_mask=mask)  # each query attends over every retrieved row
    tokens = self.action_proj(torch.cat([attended, channels], dim=-1))
    action, is_pad = self.head(tokens, state[:, None])
    return action, is_pad.squeeze(-1)

  @staticmethod
  def orthogonalize(vectors):
    """Gram-Schmidt across the query axis of (B, queries, D): each query loses its projection onto the ones before it."""
    basis = []
    for vector in vectors.unbind(dim=1):
      orthogonal = vector
      for previous in basis: orthogonal = orthogonal - (vector * previous).sum(-1, keepdim=True) / (previous * previous).sum(-1, keepdim=True) * previous
      basis.append(orthogonal)
    return torch.stack(basis, dim=1)
