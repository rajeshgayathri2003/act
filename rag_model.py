import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
import sys
from pathlib import Path
import gymnasium as gym
import mani_skill.envs
from torchvision.models import vit_b_16, vit_l_16, ViT_B_16_Weights, ViT_L_16_Weights
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import tests.config as base_config
from transformers.modeling_utils import PreTrainedModel
from faiss.rag import retrieve_sample, flatten_emb
from faiss.rag import DatasetRetriever
from datasets import load_from_disk
from detr.models.transformer import Transformer, RAGActionTransformer
import math
from dataclasses import dataclass
import argparse
import os

from transformers import TrainerCallback
from transformers import Trainer, TrainingArguments

from transformers.utils import (
    ModelOutput)

import random


def add_gt_val(k):
    """append gt with probability (1 - k).

    Args:
        k (float): Progress completed. If 0 <= k <= 1 it's treated as fraction
            (e.g. 0.35 for 35%). If 0 <= k <= 100 it's treated as percentage
            (e.g. 35 for 35%). Values outside these ranges are clipped.

    Returns:
        bool: True if gt was appended, False otherwise.
    """
    # normalize k to fraction in [0,1]
    try:
        k_val = float(k)
    except Exception:
        k_val = 0.0

    if k_val < 0:
        k_val = 0.0
    # treat values >1 and <=100 as percentages
    if 1 < k_val <= 100:
        k_frac = k_val / 100.0
    else:
        k_frac = k_val

    if k_frac < 0:
        k_frac = 0.0
    if k_frac > 1:
        k_frac = 1.0

    prob_print = max(0.0, min(1.0, 1.0 - k_frac))

    val = random.random()
    
    if val < prob_print:
        return True
    return False

sys.path.append(str(Path(__file__).parent.parent))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class ActionConfig(PretrainedConfig):
    model_type = "action_predictor"
    def __init__(
        self,
        input_size_im: int = 768,
        hidden_size_1_im: int = 512,
        hidden_size_2_im: int = 256,
        hidden_size_3_im: int = 128,
        encoder_layers: int = 2,
        out_size_im: int = 64,
        input_size_st: int = 9,
        hidden_size_st: int = 32,
        out_size_st: int = 64,
        input_size_ac: int = 201,
        hidden_size_1_ac: int = 128,
        hidden_size_2_ac: int = 128,
        hidden_size_3_ac: int = 64,
        hidden_size_4_ac: int = 16,
        out_layers_ac: int = 8,
        **kwargs
    ):
        self.input_size_im = input_size_im
        self.hidden_size_im_1 = hidden_size_1_im
        self.hidden_size_im_2 = hidden_size_2_im
        self.hidden_size_im_3 = hidden_size_3_im
        self.out_size_im = out_size_im
        self.encoder_layers = encoder_layers
        self.input_size_st = input_size_st
        self.hidden_size_st = hidden_size_st
        self.out_size_st = out_size_st
        self.input_size_ac = input_size_ac
        self.hidden_size_1_ac = hidden_size_1_ac
        self.hidden_size_2_ac = hidden_size_2_ac
        self.hidden_size_3_ac = hidden_size_3_ac
        self.hidden_size_4_ac = hidden_size_4_ac
        self.out_layers_ac = out_layers_ac
        super().__init__(**kwargs)

class ImagePreprocessor(nn.Module):
    def __init__(self, config: ActionConfig):
        super().__init__()
        self.config = config
        encoder_layers = [nn.Linear(config.input_size_im, config.hidden_size_im_1), nn.ReLU(),
                          nn.Linear(config.hidden_size_im_1, config.hidden_size_im_2), nn.ReLU(),
                          nn.Linear(config.hidden_size_im_2, config.hidden_size_im_3), nn.ReLU()]
        for _ in range(config.encoder_layers):
            encoder_layers.append(nn.Linear(config.hidden_size_im_3, config.hidden_size_im_3))
            encoder_layers.append(nn.ReLU())
        
        encoder_layers.append(nn.Linear(config.hidden_size_im_3, config.out_size_im))
        self.encoder = nn.Sequential(*encoder_layers)
        
    def forward(self, x):
        x = self.encoder(x)
        return x
    
class StatePreprocessor(nn.Module):
    def __init__(self, config: ActionConfig):
        super().__init__()
        self.config = config
        encoder_layers = [nn.Linear(config.input_size_st, config.hidden_size_st), 
                          nn.ReLU(),
                          ]
        for _ in range(config.encoder_layers):
            encoder_layers.append(nn.Linear(config.hidden_size_st, config.hidden_size_st))
            encoder_layers.append(nn.ReLU())
            
        encoder_layers.append(nn.Linear(config.hidden_size_st, config.out_size_st))
        self.encoder = nn.Sequential(*encoder_layers)
        
    def forward(self, x):
        x = self.encoder(x)
        return x
    
class ActionPreprocessor(nn.Module):
    def __init__(self, config: ActionConfig):
        super().__init__()
        self.config = config
        encoder_layers = [nn.Linear(config.input_size_ac, config.hidden_size_1_ac), 
                          nn.ReLU(),
                          nn.Linear(config.hidden_size_1_ac, config.hidden_size_2_ac),
                          nn.ReLU(),
                          nn.Linear(config.hidden_size_2_ac, config.hidden_size_3_ac),
                          nn.ReLU(),
                          nn.Linear(config.hidden_size_3_ac, config.hidden_size_4_ac),
                          nn.ReLU(),
                          nn.Linear(config.hidden_size_4_ac, config.out_layers_ac*20)
                          ]
        self.encoder = nn.Sequential(*encoder_layers)
        
    def forward(self, x):
        x = self.encoder(x)
        return x
    
class Channel(nn.Module):
    def __init__(self, input_size: int = 128, hidden_size: int = 64):
        super().__init__()

        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)
        #Modify architecture as needed
        
    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x
    
@dataclass
class ActionModelOutput(ModelOutput):
    loss: torch.FloatTensor = None
    mse_loss: torch.FloatTensor = None
    
class ActionCallback(TrainerCallback):
    def __init__(self):
        self.counter = 0
        self.mse_loss = 0
        self.add_gt = True  # boolean flag to include GT values

    def on_optimizer_step(self, args, state, control, **kwargs):
        model = trainer.model
        lr = trainer.args.learning_rate

    def on_step_end(self, args, state, control, **kwargs):
        """Update add_gt flag at every training step based on progress."""
        try:
            gs = getattr(state, "global_step", None)
            ms = getattr(state, "max_steps", None)
            total_steps = None
            if ms and ms > 0:
                total_steps = ms
            else:
                # fallback: use trainer args and dataloader length if available
                try:
                    total_steps = int(trainer.args.num_train_epochs) * len(trainer.get_train_dataloader())
                except Exception:
                    total_steps = None

            if gs is not None and total_steps:
                k_frac = float(gs) / float(total_steps)
                # update callback's add_gt based on completion fraction
                self.add_gt = add_gt_val(k_frac)
                # also update the model's add_gt attribute
                model = trainer.model
                if hasattr(model, 'add_gt'):
                    model.add_gt = self.add_gt
        except Exception:
            pass
    
    def on_log(self, args, state, control, **kwargs):
        logs = kwargs.get("logs")
        if logs is not None:
            avg_mse = self.mse_loss / self.counter if self.counter > 0 else 0
            logs["mse_loss"] = avg_mse
            # Also set eval_mse_loss for the scheduler during evaluation
            logs["eval_mse_loss"] = avg_mse
        # self.counter = 0
        # self.mse_loss = 0

class ActionModel(PreTrainedModel):
    r"""
    ActionModel is a model that predicts actions from images and states.
    """

    config_class = ActionConfig
    base_model_prefix = "action_model"

    def __init__(self, config: ActionConfig):
        super().__init__(config)
        self.image_preprocessor_query = ImagePreprocessor(config)
        self.state_preprocessor_query = StatePreprocessor(config)
        self.image_preprocessor_value = ImagePreprocessor(config)
        self.state_preprocessor_value = StatePreprocessor(config)
        self.sas_value = nn.Linear(9+4+9, 64)
        self.action_preprocessor = ActionPreprocessor(config)
        self.channels = nn.ModuleList(Channel(input_size=config.out_size_im + config.out_size_st) for i in range(3))
        self.query_fetch = nn.Linear(128, 768)
        self.query_proj = nn.Linear(64, 128)
        self.key_proj = nn.Linear(128, 768)
        self.action_proj = nn.Linear(192+64, 64)

        # encoder-decoder transformer that maps the RAG latent tokens to an action chunk
        transformer = Transformer(
            d_model=256, nhead=8,
            num_encoder_layers=4, num_decoder_layers=4,
            dim_feedforward=1024, dropout=0.1,
            normalize_before=False,
            return_intermediate_dec=False,  # so [0] is the last decoder layer
        )
        self.act_head = RAGActionTransformer(
            transformer, rag_dim=64, state_dim=9, n_rag_tokens=3,
            action_dim=4, num_queries=20,
        )
        self.pad_loss_weight = 1.0  # weight of the is_pad BCE loss relative to the action L1 loss
        # self.gt_action_key = nn.Linear(8, 128)
        # self.gt_action_proj = nn.Linear(128, 192)

        # These are loaded lazily via init_external() to avoid issues with from_pretrained
        self.processor = None
        self.clip_model = None
        self.dataset = None
        self.dataset_retriever = None
        self.T = 20
        self.add_gt = True  # boolean flag: whether to include GT values based on training progress

    def init_external(self):
        """Load CLIP model, dataset, and FAISS index. Call after from_pretrained or before training."""
        if self.dataset_retriever is not None:
            return  # already initialized
        self.processor = AutoProcessor.from_pretrained(base_config.CLIP_MODEL, use_fast=True)
        self.clip_model = AutoModel.from_pretrained(
            base_config.CLIP_MODEL,
            attn_implementation="sdpa",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            cache_dir=base_config.CACHE_DIR,
            ).to(device)
        
        #generate values
        self.dataset = load_from_disk(base_config.DEMONSTRATIONS_DIR_EE)
        self.dataset = self.dataset.map(flatten_emb)
        print("Dataset flattened")
        self.dataset_retriever = DatasetRetriever(self.dataset, 768)
        print("Dataset retriever initialized")
    def process_image(self, img):
        image_input = self.processor(
                images=img,
                return_tensors="pt",
            )
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**image_input.to(device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.view(-1).cpu().view(1, -1)
            
        return image_features
    
    def gram_schmidt(self, vectors):
        # vectors: (3, B, N)
        basis = []
        for i in range(vectors.shape[0]):
            v = vectors[i]  # (B, N)
            w = v.clone()
            
            for b in basis:
                proj_num = (v * b).sum(dim=-1, keepdim=True)  # (B, 1)
                proj_den = (b * b).sum(dim=-1, keepdim=True)  # (B, 1)
                w = w - (proj_num / proj_den) * b
            
            basis.append(w)
    
        return torch.stack(basis, dim=0)  # (3, B, N)
    
    def generate_latent(self, img, state, next_state, action_gt, return_tokens=False):
        
        image_features = self.image_preprocessor_query(img)
        state_features = self.state_preprocessor_query(state)

        #combine image and state
        combined = torch.cat((image_features, state_features), dim=-1)
        
        #Pass through three channels
        q_out = [self.channels[i](combined) for i in range(3)]
        q_stack = torch.stack(q_out, dim=0)
        q_stack = self.gram_schmidt(q_stack) 
        q_stack = q_stack.squeeze(2).permute(1, 0, 2)  # (B, 3, 64)
        q_proj = self.query_proj(q_stack)
        q_fetch = self.query_fetch(q_proj)  # (B, 3, 768)
        
        #retrieve gt action and using image as the key
        num_neighbors = 6
        retrieved_val = self.dataset_retriever.get_kv(q_fetch, neighbor_count=6)
        all_states = retrieved_val[0]  
        all_img_embs = retrieved_val[1] 
        all_actions = retrieved_val[2][:, :, 0, :]
        all_next_states = retrieved_val[3]
        
        img_key = self.image_preprocessor_query(all_img_embs.to(device))
        state_key = self.state_preprocessor_query(all_states.to(device))
    
        key = torch.cat([img_key, state_key], dim=-1) #(9, 128)
        key_gt = torch.cat([image_features, state_features], dim=-1) #(1, 128)
        
        if self.add_gt:
            key = torch.cat([key_gt, key], dim=1) #(10, 128)
        
        sas_t = torch.cat((state, action_gt[:, 0, :].unsqueeze(1), next_state), dim=-1)
        
        sas = torch.cat((all_states, all_actions, all_next_states), dim=-1)

        if self.add_gt:
            sas = torch.cat((sas_t, sas), dim=1)  # (B, 9+1, 9+8+9)
            
        sas_value = self.sas_value(sas) # (B, 10, 64)
        
        value = torch.cat([sas_value, key], dim=-1) #(64+128)

        #calculate the number of heads for multi-head attention
        B, L, D = q_proj.shape
        q = q_proj.unsqueeze(1)  # (B, 1, L, 768)
        k = key.unsqueeze(1)     # (B, 1, L, 768)
        v = value.unsqueeze(1)   # (B, 1, L, 192)
        attn_output = F.scaled_dot_product_attention(q, k, v)  # (B, 1, L, 192)
        
        attn_output = attn_output.squeeze(1)  # (B, L, 192)

        # Concatenate attention output with query and project to action dimension
        output = torch.cat([attn_output, q_stack], dim=-1)  # (B, L, 192 + 64)
        # print(output.shape)
        output = self.action_proj(output)  # (B, L, 64)
        
        if return_tokens:
            return output, state
        
        # print(output.shape, state.shape)
        output = torch.cat([output.reshape(output.shape[0], -1), state.squeeze(1)], dim=-1)  # (B,  64*3+9)
        
        return output
    
    def generate_latent_predict(self, img, state, return_tokens=False):
        
        image_features = self.image_preprocessor_query(img)
        state_features = self.state_preprocessor_query(state)

        #combine image and state
        combined = torch.cat((image_features, state_features), dim=-1)
        
        #Pass through three channels
        q_out = [self.channels[i](combined) for i in range(3)]
        q_stack = torch.stack(q_out, dim=0)
        q_stack = self.gram_schmidt(q_stack) 
        q_stack = q_stack.squeeze(2).permute(1, 0, 2)  # (B, 3, 64)
        q_proj = self.query_proj(q_stack)
        q_fetch = self.query_fetch(q_proj)  # (B, 3, 768)
        
        #retrieve gt action and using image as the key
        num_neighbors = 6
        retrieved_val = self.dataset_retriever.get_kv(q_fetch, neighbor_count=6)
        all_states = retrieved_val[0]  
        all_img_embs = retrieved_val[1] 
        all_actions = retrieved_val[2][:, :, 0, :]
        all_next_states = retrieved_val[3]
         
        img_key = self.image_preprocessor_query(all_img_embs.to(device))
        state_key = self.state_preprocessor_query(all_states.to(device))
    
        key = torch.cat([img_key, state_key], dim=-1) #(9, 128)
        key_gt = torch.cat([image_features, state_features], dim=-1) #(1, 128)
        
        sas = torch.cat((all_states, all_actions, all_next_states), dim=-1)
            
        sas_value = self.sas_value(sas) # (B, 10, 64)
        
        value = torch.cat([sas_value, key], dim=-1) #(64+128)
        
        B, L, D = q_proj.shape
        q = q_proj.unsqueeze(1)  # (B, 1, L, 768)
        k = key.unsqueeze(1)     # (B, 1, L, 768)
        v = value.unsqueeze(1)   # (B, 1, L, 192)
        attn_output = F.scaled_dot_product_attention(q, k, v)  # (B, 1, L, 192)
        
        attn_output = attn_output.squeeze(1)  # (B, L, 192)
        
        # Concatenate attention output with query and project to action dimension
        output = torch.cat([attn_output, q_stack], dim=-1)  # (B, L, 192 + 64)
        output = self.action_proj(output)  # (B, L, 64)

        if return_tokens:
            return output, state

        output = torch.cat([output.reshape(output.shape[0], -1), state.squeeze(1)], dim=-1)  # (B,  64*3+9)
        
        return output
        
    def forward(self, img, state, next_state, action_gt, is_pad=None, **kwargs):
        img = img.to(device)
        state = state.to(device)
        next_state = next_state.to(device)
        action_gt = action_gt.to(device)

        rag_tokens, state_tok = self.generate_latent(img, state, next_state, action_gt, return_tokens=True)
        # rag_tokens: (B, 3, 64), state_tok: (B, 1, 9)

        action_pred, is_pad_hat = self.act_head(rag_tokens, state_tok)  # (B, T, 4), (B, T, 1)
        #loss
        loss = None
        l1 = None
        if action_gt is not None:
            if is_pad is None:
                is_pad = torch.zeros(action_gt.shape[:2], dtype=torch.bool)
            is_pad = is_pad.to(device)
            #changed from mse to l1 loss here, only over the real (non-padded) timesteps
            valid = (~is_pad).unsqueeze(-1).to(action_pred.dtype)  # (B, T, 1)
            all_l1 = F.l1_loss(action_pred, action_gt.to(action_pred.dtype), reduction='none')
            l1 = (all_l1 * valid).sum() / (valid.sum() * action_pred.shape[-1]).clamp(min=1)
            #train is_pad_head to predict which timesteps of the chunk are padding
            pad_loss = F.binary_cross_entropy_with_logits(is_pad_hat.squeeze(-1), is_pad.to(is_pad_hat.dtype))
            loss = l1 + self.pad_loss_weight * pad_loss

        return ActionModelOutput(loss=loss, mse_loss=l1.detach() if l1 is not None else 0.0)

    @torch.no_grad()
    def predict(self, img, state):
        rag_tokens, state_tok = self.generate_latent_predict(img.to(device), state.to(device), return_tokens=True)
        action_pred, _ = self.act_head(rag_tokens, state_tok)
        return action_pred  # (B, T, 4)

def data_collator(batch):
        # Stack the image and text embeddings
        images = torch.stack([torch.tensor(x['image_emb']) for x in batch])
        # print(images.shape)
        # texts = torch.stack([torch.tensor(x['text_emb']) for x in batch])
        state = torch.stack([torch.tensor(x['state']) for x in batch])
        action = torch.stack([torch.tensor(x['action']) for x in batch])
        next_state = torch.stack([torch.tensor(x['next_state']) for x in batch])
        # Action chunks are zero-padded past the episode end (see data_encoder.py), so padded steps
        # are the trailing all-zero rows. Use a stored 'is_pad' column instead if the dataset has one.
        if 'is_pad' in batch[0]:
            is_pad = torch.stack([torch.tensor(x['is_pad'], dtype=torch.bool) for x in batch])
        else:
            nonzero = (action != 0).any(dim=-1)  # (B, T)
            is_pad = nonzero.flip(-1).int().cummax(dim=-1).values.flip(-1) == 0

        return {'img':images, 'state':state,'action_gt': action, 'next_state': next_state, 'is_pad': is_pad, 'labels': ""}
    
if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--log_dir", type=str, default=base_config.ACTION_LOGDIR, help="Directory of the for logging the training metrics and losses")
    args.add_argument("--model_dir", type=str, default=base_config.ACTION_MODEL, help="Directory of the model checkpoints")
    args.add_argument("--resume_from_checkpoint", action="store_true", help="Whether training should be resumed from a checkpoint")
    args = args.parse_args()
    
    logging_dir = base_config.ACTION_LOGDIR.format(args.log_dir)
    model_dir = base_config.ACTION_MODEL.format(args.model_dir)
    
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    
    #in ee action mode the number of actions is 4
    action_config = ActionConfig(out_layers_ac=4)  
    model = ActionModel(action_config)
    model.init_external()
    model.to(device)
    
    trainer_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=base_config.ACTION_TRAIN_EPOCHS,
        per_device_train_batch_size=base_config.ACTION_BATCH_SIZE,
        save_steps=base_config.ACTION_SAVE_STEPS,
        save_total_limit=base_config.ACTION_SAVE_LIMIT,
        logging_dir=logging_dir,
        logging_steps=base_config.ACTION_LOG_STEPS,
        report_to="tensorboard",
        learning_rate=base_config.ACTION_LR,
        lr_scheduler_type="reduce_lr_on_plateau",
        lr_scheduler_kwargs={
            "factor": 0.1,      # multiply LR by this on plateau
            "patience": 2,      # epochs without improvement before reducing
            "threshold": 1e-3,  # min change to count as improvement
        },
        metric_for_best_model="mse_loss",
        remove_unused_columns=False,
        no_cuda=False,
        eval_strategy="epoch",
        greater_is_better=False
    )

    callback_object = ActionCallback()

    def compute_loss_fn(outputs, labels, num_items_in_batch):
        callback_object.counter += 1
        mse_val = outputs.mse_loss.item() if hasattr(outputs.mse_loss, 'item') else float(outputs.mse_loss)
        callback_object.mse_loss += mse_val
        # callback_object.recon_loss += outputs.recon_loss
        return outputs.loss
    
    def compute_metrics(eval_pred):
        avg = float(callback_object.mse_loss / callback_object.counter) if callback_object.counter > 0 else 0.0
        avg = float(avg)  # ensure it's a plain Python float
        callback_object.counter = 0
        callback_object.mse_loss = 0.0
        return {"mse_loss": avg}    

    dataset = load_from_disk(base_config.DEMONSTRATIONS_DIR_EE_PUSH_CUBE)
    print("Dataset loaded")
    trainer = Trainer(
        model=model,
        args=trainer_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=dataset, 
        callbacks=[callback_object],
        compute_loss_func=compute_loss_fn,
        compute_metrics=compute_metrics,
    )
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)  
    #save the model
    trainer.save_model(model_dir)
    
    