import argparse
import glob
import os
import re

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

import config as base_config
from rag_model import ActionModel

# Same image encoder used to build the training dataset (see ijepa_encoder.py)
IJEPA_MODEL_ID = "jmtzt/ijepa_vith16_1k"


def find_latest_checkpoint(model_dir):
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    if not checkpoints:
        return model_dir
    return max(checkpoints, key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))


class InferModel:
    def __init__(self, checkpoint_dir, device):
        self.device = device
        self.action_model = ActionModel.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        ).to(device)
        self.action_model.init_external()
        self.action_model.eval()

        self.img_processor = AutoProcessor.from_pretrained(IJEPA_MODEL_ID)
        self.img_model = AutoModel.from_pretrained(IJEPA_MODEL_ID).to(device).eval()

    def encode_image(self, image):
        """Encode a raw rendered frame into the IJEPA embedding space the model was trained on."""
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy().astype(np.uint8)
        inputs = self.img_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_emb = self.img_model(**inputs).last_hidden_state.mean(dim=1)  # (1, 1280)
        return image_emb.unsqueeze(1)  # (1, 1, 1280) - matches the training dataset's image_emb layout

    def infer(self, image, state):
        state = state.to(self.device).float()
        if state.dim() == 2:
            state = state.unsqueeze(1)  # (B, 1, 9) - matches the training dataset's state layout
        image_emb = self.encode_image(image)
        with torch.no_grad():
            action_pred = self.action_model.predict(image_emb, state)  # (1, T, 4)
        return action_pred

    def get_action(self, image, state):
        action_chunk = self.infer(image, state)
        action = action_chunk[0, 0]  # first step of the predicted chunk
        return action, 0.0


if __name__ == '__main__':
    import gymnasium as gym
    import imageio
    import mani_skill.envs
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="rag_act",
                         help="Suffix of the trained model directory (see config.ACTION_MODEL)")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Checkpoint subdir name to load; defaults to the latest in model_dir")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_episode_steps", type=int, default=400)
    parser.add_argument("--video_dir", type=str, default="/gayu-west/videos_rag_act")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = base_config.ACTION_MODEL.format(args.model_dir)
    checkpoint_dir = os.path.join(model_dir, args.checkpoint) if args.checkpoint else find_latest_checkpoint(model_dir)
    print(f"Loading checkpoint: {checkpoint_dir}")

    model = InferModel(checkpoint_dir, device)

    env = gym.make(
        "PushCube-v1",
        robot_uids="panda",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
    )

    os.makedirs(args.video_dir, exist_ok=True)
    rng = np.random.default_rng(1)

    successes = 0
    for idx in tqdm(range(args.num_episodes)):
        seed = rng.integers(0, 1000000)
        obs, _ = env.reset(seed=[int(seed)])
        done = False
        total_reward = 0.0
        counter = 0
        frames = []
        while not done:
            state = obs['agent']['qpos']
            image = env.render()

            frame = image.squeeze().cpu().numpy()
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            frames.append(frame)

            action, _ = model.get_action(image, state)
            obs, reward, terminated, truncated, info = env.step(action.cpu())
            done = terminated or truncated
            total_reward += reward
            counter += 1

        video_path = os.path.join(args.video_dir, f"episode_{idx}.mp4")
        imageio.mimwrite(video_path, frames, fps=20, quality=8)

        success = bool(terminated)
        successes += int(success)
        print(f"Episode {idx}: Total Reward {float(total_reward) / counter:.4f} Success {success} Timestep {counter}")

    print(f"Success rate: {successes}/{args.num_episodes}")
