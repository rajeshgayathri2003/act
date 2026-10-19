"""Image encoders that turn rendered frames into the embeddings that retrieval and the models consume.

Subclasses differ only in the vision model and how its output pools to one vector, so encoders compare on equal terms.
"""
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel


class Encoder(ABC):
  """Frames in, one embedding per frame out: (B, H, W, 3) uint8 to (B, dim) float32 on the CPU."""

  model = AutoModel  # transformers class that loads the vision weights; the instance replaces it in __init__
  model_id: str  # the weights on the Hugging Face hub

  def __init__(self, config, device="cpu"):
    self.config = config
    self.device = device
    self.processor = AutoImageProcessor.from_pretrained(self.model_id)  # image side only, so SigLIP needs no tokenizer
    self.model = self.model.from_pretrained(self.model_id).to(device).eval()
    self.dim = self.model.config.hidden_size

  @abstractmethod
  def pool(self, outputs):
    """Reduce the vision model's outputs to (B, dim)."""

  @torch.inference_mode()
  def embed(self, images):
    inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
    with torch.autocast(torch.device(self.device).type, dtype=getattr(torch, self.config.precision), enabled=self.config.precision != "float32"):
      embeddings = self.pool(self.model(**inputs)).float()
    if self.config.normalize: embeddings = F.normalize(embeddings, dim=-1)  # retrieval searches by inner product
    return embeddings.cpu()


class SiglipEncoder(Encoder):
  """SigLIP's attention-pooled image embedding, the same vector as SiglipModel.get_image_features."""

  model = SiglipVisionModel  # vision tower only; the text tower is never used
  model_id = "google/siglip-base-patch16-224"

  def pool(self, outputs): return outputs.pooler_output


class IjepaEncoder(Encoder):
  """I-JEPA patch tokens averaged into one vector; the model has no pooled output of its own."""

  model_id = "jmtzt/ijepa_vith16_1k"

  def pool(self, outputs): return outputs.last_hidden_state.mean(dim=1)


ENCODERS = {
  "siglip": SiglipEncoder,
  "ijepa": IjepaEncoder,
}
