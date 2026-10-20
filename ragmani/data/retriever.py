"""Nearest-neighbour lookup over the encoded demos: the memory the RAG models attend to.

Exact search is a matmul on whatever device the model runs on; FAISS IVF is the option for datasets too large for that.
Choosing which neighbours to use (subsampling, how many) belongs to the caller, so the retriever returns all k.
"""
import torch


class Retriever:
  """Holds the encoded rows on one device and finds the nearest by inner product of image embeddings."""

  COLUMNS = ("image_emb", "state", "action", "next_state", "episode", "task", "demo", "step")  # task, demo and step locate a row to render it

  def __init__(self, dataset, config, device="cpu"):
    self.config = config
    rows = dataset.with_format("torch", columns=list(self.COLUMNS))[:]
    self.longest_episode = int(torch.bincount(rows["episode"]).max())  # IVF over-fetches this many so exclusion still leaves k
    self.rows = {key: value.to(device) for key, value in rows.items()}
    self.on_gpu = False  # build sets it once FAISS moves the index to the GPU
    self.index = self.build() if config.index == "ivf" else None
    print(self.describe())

  def build(self):
    """FAISS IVF with 8-bit scalar quantization, on the GPU when FAISS has GPU support (faiss-gpu is x86_64 only, so not on sparky)."""
    import faiss  # only here: faiss-cpu and torch each bundle libomp on macOS, and loading both aborts the process

    embeddings = self.rows["image_emb"].float().cpu().numpy()
    D = embeddings.shape[1]
    nlist = self.config.nlist or int(len(embeddings) ** 0.5)
    index = faiss.IndexIVFScalarQuantizer(faiss.IndexFlatIP(D), D, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = self.config.nprobe

    if self.rows["image_emb"].is_cuda and hasattr(faiss, "StandardGpuResources"):
      import faiss.contrib.torch_utils  # lets index.search take and return torch tensors, CUDA ones without a copy

      self.gpu_resources = faiss.StandardGpuResources()  # must outlive the index
      index = faiss.index_cpu_to_gpu(self.gpu_resources, self.rows["image_emb"].device.index or 0, index)
      self.on_gpu = True
    return index

  def describe(self):
    """One line on how this memory searches, printed when it is built; add anything worth seeing in a run's log here."""
    where = "FAISS on the GPU" if self.on_gpu else "FAISS on the CPU" if self.index is not None else f"exact on {self.rows['image_emb'].device}"
    ivf = f", nlist {self.index.nlist}, nprobe {self.index.nprobe}" if self.index is not None else ""
    return f"retriever: {len(self.rows['episode'])} rows, {where}{ivf}, longest episode {self.longest_episode}"

  @torch.no_grad()
  def search(self, query, k, exclude=None):
    """(B, Q, D) query to (B, Q, k) row indices, best first; exclude is (B,) episode ids never returned for that row."""
    if self.index is None:  # if not using FAISS, do exact search
      scores = query @ self.rows["image_emb"].T  # (B, Q, N)
      if exclude is not None: scores.masked_fill_((self.rows["episode"] == exclude[:, None])[:, None, :], float("-inf"))
      return scores.topk(k, dim=-1).indices

    B, Q, D = query.shape
    fetch = k if exclude is None else k + self.longest_episode
    search = self.search_gpu if self.on_gpu else self.search_cpu
    found = search(query.reshape(-1, D).float(), fetch).reshape(B, Q, fetch)
    keep = found >= 0  # IVF returns -1 when the probed cells hold fewer than fetch rows
    if exclude is not None: keep &= self.rows["episode"][found.clamp(min=0)] != exclude[:, None, None]
    first = (~keep).int().argsort(dim=-1, stable=True)[..., :k]  # kept rows first, in score order
    return found.gather(-1, first)

  def search_cpu(self, query, fetch):
    """(B*Q, D) query to (B*Q, fetch) row indices on query's device, through a FAISS index in CPU memory."""
    _, found = self.index.search(query.cpu().numpy(), fetch)
    return torch.from_numpy(found).to(query.device)

  def search_gpu(self, query, fetch):
    """(B*Q, D) query to (B*Q, fetch) row indices, staying on the GPU with the index; GPU FAISS allows fetch up to 2048."""
    _, found = self.index.search(query.contiguous(), fetch)
    return found

  def gather(self, indices):
    """Every column at the given row indices, each shaped indices.shape + the column's own shape."""
    return {key: value[indices] for key, value in self.rows.items()}
