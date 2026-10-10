"""Nearest-neighbour lookup over the encoded demos: the memory the RAG models attend to.

Exact search is a matmul on whatever device the model runs on; FAISS IVF is the option for datasets too large for that.
Choosing which neighbours to use (subsampling, how many) belongs to the caller, so the retriever returns all k.
"""
import torch

COLUMNS = ["image_emb", "state", "action", "next_state", "episode"]


class Retriever:
  """Holds the encoded rows on one device and finds the nearest by inner product of image embeddings."""

  def __init__(self, dataset, config, device="cpu"):
    self.config = config
    rows = dataset.with_format("torch", columns=COLUMNS)[:]
    self.longest_episode = int(torch.bincount(rows["episode"]).max())  # IVF over-fetches this many so exclusion still leaves k
    self.rows = {key: value.to(device) for key, value in rows.items()}
    self.index = self.build() if config.index == "ivf" else None

  def build(self):
    """FAISS IVF with 8-bit scalar quantization, on the GPU when FAISS has GPU support."""
    import faiss  # only here: faiss-cpu and torch each bundle libomp on macOS, and loading both aborts the process

    embeddings = self.rows["image_emb"].float().cpu().numpy()
    dim = embeddings.shape[1]
    nlist = self.config.nlist or int(len(embeddings) ** 0.5)
    index = faiss.IndexIVFScalarQuantizer(faiss.IndexFlatIP(dim), dim, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = self.config.nprobe

    if self.rows["image_emb"].is_cuda and hasattr(faiss, "StandardGpuResources"):
      self.gpu_resources = faiss.StandardGpuResources()  # must outlive the index
      index = faiss.index_cpu_to_gpu(self.gpu_resources, self.rows["image_emb"].device.index or 0, index)
    return index

  @torch.no_grad()
  def search(self, queries, k, exclude=None):
    """(B, Q, dim) queries to (B, Q, k) row indices, best first; exclude is (B,) episode ids never returned for that row."""
    if self.index is None:
      scores = queries @ self.rows["image_emb"].T  # (B, Q, N)
      if exclude is not None: scores.masked_fill_((self.rows["episode"] == exclude[:, None])[:, None, :], float("-inf"))
      return scores.topk(k, dim=-1).indices

    batch, count, dim = queries.shape
    fetch = k if exclude is None else k + self.longest_episode
    _, found = self.index.search(queries.reshape(-1, dim).float().cpu().numpy(), fetch)
    found = torch.from_numpy(found).to(queries.device).reshape(batch, count, fetch)
    keep = found >= 0  # IVF returns -1 when the probed cells hold fewer than fetch rows
    if exclude is not None: keep &= self.rows["episode"][found.clamp(min=0)] != exclude[:, None, None]
    first = (~keep).int().argsort(dim=-1, stable=True)[..., :k]  # kept rows first, in score order
    return found.gather(-1, first)

  def gather(self, indices):
    """Every column at the given row indices, each shaped indices.shape + the column's own shape."""
    return {key: value[indices] for key, value in self.rows.items()}
