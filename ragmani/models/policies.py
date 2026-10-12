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


KEEP_SAMPLING = {  # shape of the random numbers that choose which neighbours to keep
  "per_row": lambda shape: shape,          # separately for every row and search
  "per_batch": lambda shape: shape[-1:],   # the same positions for the whole batch, as rag_model.py's randperm
}

RAG_MODEL_RENAMES = [  # rag_model.py ActionModel weight names to RagAct's
  (r"^image_preprocessor_query\.encoder\.", "image_encoder."),
  (r"^state_preprocessor_query\.encoder\.", "state_encoder."),
  (r"^sas_value\.", "transition_value."),
  (r"^channels\.(\d+)\.linear_(\d)\.", lambda match: f"channels.{match[1]}.{int(match[2]) - 1}."),
  (r"^act_head\.", "head."),
]
RAG_MODEL_UNUSED = ("image_preprocessor_value.", "state_preprocessor_value.", "action_preprocessor.", "key_proj.")


def standardize(scale, value):
  """value scaled by a (2, dim) scale of mean and std."""
  return (value - scale[0]) / scale[1]


def mlp(sizes):
  """Linear+ReLU between consecutive sizes, with no ReLU after the last Linear."""
  layers = [module for width_in, width_out in zip(sizes[:-2], sizes[1:-1]) for module in (nn.Linear(width_in, width_out), nn.ReLU())]
  return nn.Sequential(*layers, nn.Linear(sizes[-2], sizes[-1]))


class RagAct(nn.Module):
  """Retrieval-augmented ACT: three orthogonal queries fetch demo rows, attention turns them into tokens, an ACT head decodes the chunk.

  Gayathri Rajesh's ActionModel (rag_model.py) restructured: the same layers and sizes, minus four it built and never used.
  """

  def __init__(self, config, image_dim):
    super().__init__()
    self.config = config
    features = config.feature_size  # image and state are each reduced to this many; a key joins the two
    self.image_encoder = mlp([image_dim, *config.image_layers, features])
    self.state_encoder = mlp([config.state_dim, *config.state_layers, features])
    self.transition_value = nn.Linear(2 * config.state_dim + config.action_dim, config.token_dim)  # (state, first action, next_state) of a row
    self.channels = nn.ModuleList(nn.Sequential(nn.Linear(2 * features, config.channel_hidden), nn.Linear(config.channel_hidden, config.channel_hidden), nn.Linear(config.channel_hidden, config.channel_hidden)) for _ in range(config.queries))  # no activations, as in the original
    self.query_fetch = nn.Linear(2 * features, image_dim) if config.query == "learned" else nn.Identity()  # into the image embedding's space; the search passes no gradient, so it trains only through search_scores_in_attention
    self.query_proj = nn.Linear(config.channel_hidden, 2 * features)
    self.action_proj = nn.Linear(config.token_dim + 2 * features + config.channel_hidden, config.token_dim)  # what attention read from the retrieved rows, plus the search's own features, mixed into one token for the ACT head
    transformer = Transformer(
      d_model=config.d_model, nhead=config.heads, num_encoder_layers=config.encoder_layers, num_decoder_layers=config.decoder_layers,
      dim_feedforward=config.feedforward, dropout=config.dropout, normalize_before=False, return_intermediate_dec=False,
    )
    self.head = RAGActionTransformer(transformer, rag_dim=config.token_dim, state_dim=config.state_dim, n_rag_tokens=config.queries, action_dim=config.action_dim, num_queries=config.chunk)
    self.register_buffer("state_scale", torch.stack([torch.zeros(config.state_dim), torch.ones(config.state_dim)]))  # mean and std; identity unless training sets them
    self.register_buffer("action_scale", torch.stack([torch.zeros(config.action_dim), torch.ones(config.action_dim)]))

  @classmethod
  def load(cls, checkpoint_dir, config, image_dim):
    """A checkpoint saved by ragmani's trainer, or failing that one saved by rag_model.py."""
    model = cls(config, image_dim)
    try: model.load_state_dict({**dict(model.named_buffers()), **load_file(Path(checkpoint_dir) / "model.safetensors")})  # checkpoints from before the scales load as unscaled
    except (RuntimeError, FileNotFoundError) as error:
      try: return cls.from_rag_model(checkpoint_dir, config)
      except Exception as fallback: raise error from fallback  # not a rag_model.py checkpoint either; the first failure explains it, the second is chained
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
    model.load_state_dict({**dict(model.named_buffers()), **renamed})
    return model

  def forward(self, image, state, retriever, exclude=None, truth=None, add_truth=None):
    """Standardized action chunk (B, chunk, action_dim), padding logits (B, chunk) and the retrieved row indices (B, queries*keep).

    image is (B, image_dim) and state (B, state_dim), both raw. exclude is (B,) episode ids the search skips. truth is each
    row's own raw (action chunk, next_state), added as one more key and value for the rows where add_truth (B,) is True.
    """
    state = standardize(self.state_scale, state)
    own_key = torch.cat([self.image_encoder(image), self.state_encoder(state)], dim=-1)  # (B, 2*features)
    channels = self.orthogonalize(torch.stack([channel(own_key) for channel in self.channels], dim=1))  # (B, queries, channel_hidden)
    queries = self.query_proj(channels)

    fetch = self.query_fetch(queries if self.config.query == "learned" else image[:, None].expand(-1, self.config.queries, -1))  # observation searches with the frame's own embedding
    found = retriever.search(fetch, self.config.neighbours, exclude)  # (B, queries, neighbours)
    kept = torch.rand(KEEP_SAMPLING[self.config.keep_sampling](found.shape), device=found.device).argsort(dim=-1)[..., :self.config.keep].expand(*found.shape[:-1], self.config.keep)
    indices = found.gather(-1, kept).flatten(1)  # every query's rows together
    rows = retriever.gather(indices)
    rows_state, rows_next_state = standardize(self.state_scale, rows["state"]), standardize(self.state_scale, rows["next_state"])
    keys = torch.cat([self.image_encoder(rows["image_emb"]), self.state_encoder(rows_state)], dim=-1)
    transitions = torch.cat([rows_state, standardize(self.action_scale, rows["action"][:, :, 0]), rows_next_state], dim=-1)

    images = rows["image_emb"]
    if truth is not None:
      action, next_state = truth
      keys, images = torch.cat([own_key[:, None], keys], dim=1), torch.cat([image[:, None], images], dim=1)
      transitions = torch.cat([torch.cat([state, standardize(self.action_scale, action[:, 0]), standardize(self.state_scale, next_state)], dim=-1)[:, None], transitions], dim=1)

    bias = fetch @ images.transpose(1, 2) if self.config.search_scores_in_attention else keys.new_zeros(keys.shape[0], 1, keys.shape[1])  # (B, queries or 1, keys): each row's search similarity, added to its attention weight
    if truth is not None: bias = bias.masked_fill(F.pad(~add_truth[:, None, None], (0, keys.shape[1] - 1)), float("-inf"))  # the truth key, only where add_truth

    values = torch.cat([self.transition_value(transitions), keys], dim=-1)
    attended = F.scaled_dot_product_attention(queries, keys, values, attn_mask=bias)  # each query attends over every retrieved row
    tokens = self.action_proj(torch.cat([attended, channels], dim=-1))
    action, is_pad = self.head(tokens, state[:, None])
    return action, is_pad.squeeze(-1), indices

  @staticmethod
  def orthogonalize(vectors):
    """Gram-Schmidt across the query axis of (B, queries, D): each query loses its projection onto the ones before it."""
    basis = []
    for vector in vectors.unbind(dim=1):
      orthogonal = vector
      for previous in basis: orthogonal = orthogonal - (vector * previous).sum(-1, keepdim=True) / (previous * previous).sum(-1, keepdim=True) * previous
      basis.append(orthogonal)
    return torch.stack(basis, dim=1)
