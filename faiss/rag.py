from typing import Tuple, Dict

import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import faiss
import random
import tests.config as base_config
import faiss.contrib.torch_utils

# import faiss.contrib.torch_utils

from abc import ABC, abstractmethod

#Method 1
from datasets import load_from_disk
import tests.config
import numpy as np
from abc import ABC, abstractmethod

class IRetriever(ABC):
    @abstractmethod
    def get_kv(self, queries: torch.Tensor, neighbor_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def add(self, text, image, emb_text, emb_image):
        pass
    
class DatasetRetriever(IRetriever):
    def __init__(self, dataset, emb_dim, gpu_id=0):
        self.res = faiss.StandardGpuResources()

        print("Caching dataset columns in memory...")
        self.image_emb = np.vstack(dataset['image_emb']).astype(np.float32)
        self.state = np.vstack(dataset['state']).astype(np.float32)
        self.action = np.vstack(dataset['action']).reshape(len(dataset), -1).astype(np.float32)
        self.next_state = np.vstack(dataset['next_state']).astype(np.float32)
        print(f"Cached: image_emb {self.image_emb.shape}, state {self.state.shape}, action {self.action.shape}, next_state {self.next_state.shape}")

        d = self.image_emb.shape[1]
        n = self.image_emb.shape[0]
        
        # IVF cell count: rule of thumb is sqrt(n), power of 2
        nlist = min(4096, int(np.sqrt(n)))
        
        self.cpu_index = faiss.IndexIVFScalarQuantizer(
            faiss.IndexFlatIP(d),           # flat quantizer (safe, always works)
            d,
            nlist,
            faiss.ScalarQuantizer.QT_8bit,  # 4x compression, minimal accuracy loss
            faiss.METRIC_INNER_PRODUCT
        )
                
        print("Training index...")
        self.cpu_index.train(np.ascontiguousarray(self.image_emb))

        self.cpu_index.nprobe = 8    # cells to search at query time (higher = more accurate, slower)
        print("Adding vectors...")
        self.cpu_index.add(np.ascontiguousarray(self.image_emb))
        # self.cpu_index = faiss.IndexFlatIP(self.image_emb.shape[1])
        # self.cpu_index.add(np.ascontiguousarray(self.image_emb))
        self.index = faiss.index_cpu_to_gpu(self.res, gpu_id, self.cpu_index)

        # Pre-cache on GPU once — eliminates torch.tensor() every forward pass
        print("Moving cached arrays to GPU...")
        self.image_emb_gpu = torch.tensor(self.image_emb, dtype=torch.float32).cuda()
        self.state_gpu     = torch.tensor(self.state,     dtype=torch.float32).cuda()
        self.action_gpu    = torch.tensor(self.action,    dtype=torch.float32).cuda()
        self.next_state_gpu = torch.tensor(self.next_state, dtype=torch.float32).cuda()
        print(f"Index ready with {self.index.ntotal} vectors")

    def get_kv(self, queries, neighbor_count):
        shape = queries.shape
        q = queries.detach().view(-1, shape[-1]).contiguous()
        
        _, examples = self.index.search(q, neighbor_count) #(B*channels, neighbor_count)
        
        indices = examples.reshape(shape[0], shape[1], neighbor_count).cpu()  # small index tensor, cheap to move
        
        sample_val = neighbor_count // 2
        sample_idx = torch.randperm(neighbor_count)[:sample_val]
        
        sampled_indices = indices[:, :, sample_idx]  # (B, channels, sample_val)
        
        sampled_indices = sampled_indices.reshape(-1)  # flatten for indexing 

        # Direct GPU tensor indexing — no CPU transfer
        emb_img = self.image_emb_gpu[sampled_indices]
        state   = self.state_gpu[sampled_indices]
        action  = self.action_gpu[sampled_indices]
        next_state = self.next_state_gpu[sampled_indices]
        
        emb_img = emb_img.view(shape[0], shape[1]*sample_val, -1)
        state   = state.view(shape[0],   shape[1]*sample_val, -1)
        action  = action.view(shape[0],  shape[1]*sample_val, 20, 4)
        next_state = next_state.view(shape[0], shape[1]*sample_val, -1)
        # action  = action.view(shape[0],  shape[1]*sample_val, -1)
        return state, emb_img, action, next_state
    
    def add(self, text, image, emb_text, emb_image):
        return
    
class DatasetRetriever_Prev(IRetriever):
    def __init__(self, dataset, emb_dim, gpu_id=0):
        self.res = faiss.StandardGpuResources()

        # Pre-cache all columns as contiguous numpy arrays in memory
        # so get_kv doesn't do slow Arrow random-access reads every step
        print("Caching dataset columns in memory...")
        self.image_emb = np.vstack(dataset['image_emb']).astype(np.float32)
        self.state = np.vstack(dataset['state']).astype(np.float32)
        self.action = np.vstack(dataset['action']).astype(np.float32)
        print(f"Cached: image_emb {self.image_emb.shape}, state {self.state.shape}, action {self.action.shape}")

        self.cpu_index = faiss.IndexFlatIP(self.image_emb.shape[1])

        if len(self.image_emb) > 0:
            self.cpu_index.add(np.ascontiguousarray(self.image_emb))

        # Move to GPU after populating
        self.index = faiss.index_cpu_to_gpu(self.res, gpu_id, self.cpu_index)
        
        print(f"Index ready with {self.index.ntotal} vectors")
        
    def get_kv(self, queries, neighbor_count):
        
        shape = queries.shape
        q = queries.detach().view(-1, shape[-1]).contiguous()  # stays on GPU
        
        _, examples = self.index.search(q, neighbor_count)  # GPU tensor in, GPU tensor out
        
        indices = examples.reshape(-1).cpu().numpy()
        
        emb_img = torch.tensor(self.image_emb[indices], dtype=torch.float32).cuda()
        state   = torch.tensor(self.state[indices],     dtype=torch.float32).cuda()
        action  = torch.tensor(self.action[indices],    dtype=torch.float32).cuda()
        
        
        
        
        
        emb_img = emb_img.view(shape[0], shape[1]*neighbor_count, -1)
        state   = state.view(shape[0],   shape[1]*neighbor_count, -1)
        action  = action.view(shape[0],  shape[1]*neighbor_count, -1)
        
        return state, action, emb_img
        
    def get_kv_old(self, queries, neighbor_count):
        shape = queries.shape
        qcpu = queries.cpu().detach().view(-1, shape[-1]).numpy().astype(np.float32)
        _, examples = self.index.search(qcpu, neighbor_count)
        
        indices = examples.reshape(-1)
        
        # Fast numpy indexing on pre-cached arrays (no disk I/O)
        emb_img = torch.tensor(self.image_emb[indices], dtype=torch.float32).cuda()
        state = torch.tensor(self.state[indices], dtype=torch.float32).cuda()
        action = torch.tensor(self.action[indices], dtype=torch.float32).cuda()
        
        emb_img = emb_img.view((shape[0], shape[1], neighbor_count, -1))
        state = state.view((shape[0], shape[1], neighbor_count, -1))
        action = action.view((shape[0], shape[1], neighbor_count, -1))
        return state, action, emb_img
    
    def add(self, text, image, emb_text, emb_image):
        return


def flatten_emb(sample):
    # print(sample.keys())
    sample["image_emb"] = np.array(sample["image_emb"]).reshape(-1)
    return sample

def retrieve_sample(dataset, query, k = 1):

    dataset.add_faiss_index(column="image_emb")
    all_retrieved_examples = []
    print("query shape", query.shape)
    for q in query:
        scores, retrieved_examples = dataset.get_nearest_examples(
            "image_emb",
            q,
            k
        )
        all_retrieved_examples.append(retrieved_examples)
        
    all_states = [torch.tensor(retrieved_example['state']) for retrieved_example in all_retrieved_examples]
    all_actions = [torch.tensor(retrieved_example['action']) for retrieved_example in all_retrieved_examples]
    all_img_embs = [torch.tensor(retrieved_example['image_emb']) for retrieved_example in all_retrieved_examples]
    
    all_states = torch.stack(all_states, dim=0)
    all_actions = torch.stack(all_actions, dim=0)
    all_img_embs = torch.stack(all_img_embs, dim=0)
    
    # print(scores, retrieved_examples)
    return (all_states, all_actions, all_img_embs)

def retrieve_sample_single(dataset, query, k = 1):

    dataset.add_faiss_index(column="image_emb")

    scores, retrieved_examples = dataset.get_nearest_examples(
        "image_emb",
        query,
        k
    )
    # print(scores, retrieved_examples)
    return retrieved_examples

if __name__ == "__main__":
    dataset = load_from_disk(base_config.DEMONSTRATIONS_DIR_EE_PUSH_CUBE_JEPA)
    dataset = dataset.map(flatten_emb)
    print("Dataset flattened")
    dataset_retriever = DatasetRetriever(dataset, 1260)
    print("Dataset retriever initialized")