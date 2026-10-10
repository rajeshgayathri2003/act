import datasets
import torchvision.transforms as transforms
from detr.models.backbone import Backbone
import gymnasium as gym
import config
import mani_skill.envs
import torch
import mani_skill.trajectory.utils as trajectory_utils
from tqdm import tqdm
import h5py
import numpy as np
import argparse

#resnet weights saved under this prefix inside the ACT agent's state dict (see detr/models/backbone.py Joiner)
BACKBONE_STATE_DICT_PREFIX = "model.backbones.0.0."

def load_trained_backbone(model_img, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent_state = checkpoint["agent"]
    backbone_state = {
        k[len(BACKBONE_STATE_DICT_PREFIX):]: v
        for k, v in agent_state.items()
        if k.startswith(BACKBONE_STATE_DICT_PREFIX)
    }
    model_img.load_state_dict(backbone_state)
    return model_img

def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out

def create_env(env_id):
    print(env_id)
    env = gym.make(env_id, num_envs=1, render_mode="rgb_array", obs_mode = "rgb+depth+segmentation")
    return env

def load_env_state(env, env_states, timestep):
    env_states = trajectory_utils.dict_to_list_of_dicts(env_states)
    env_state_t = trajectory_utils.index_dict(env_states, timestep)
    env.reset()
    env.set_state_dict(env_state_t)

def process_data_from_h5(model_img, processor, device, traj_path, env_id, control_mode):
    def generator():
        env = gym.make(env_id, num_envs=1, render_mode="rgb_array", obs_mode = "rgb+depth+segmentation", control_mode=control_mode)
        print(env.single_action_space.shape)

        with h5py.File(traj_path, "r") as f:
            for traj_key in tqdm(list(f.keys())):
                env.reset()
                cur_traj = load_h5_data(f[traj_key])
                actions = cur_traj["actions"][:]
                state_dict = cur_traj["env_states"]
                T = len(actions)

                batch_images, batch_actions, batch_states, batch_next_states = [], [], [], []

                for t in range(T):
                    load_env_state(env, state_dict, t)
                    img = env.render()                         # numpy array (H, W, C)
                    obs = env.get_obs()
                    state = obs['agent']['qpos']

                    #process image before appending to batch
                    torch.cuda.empty_cache()

                    #same preprocessing as ACT: scale to [0, 1], then ImageNet normalize (policy.py)
                    img = torch.as_tensor(img, device=device)
                    if img.ndim == 3:
                        img = img.unsqueeze(0)
                    img = img.permute(0, 3, 1, 2).float() / 255.0
                    img = processor(img)
                    with torch.inference_mode():
                        features = model_img(img)["0"]  # layer4 map: (1, 512, H/32, W/32)

                    #global average pool the feature map into a 512-dim vector
                    image_emb = features.mean(dim=(2, 3)).cpu()
                    batch_images.append(image_emb)

                    #T=20 timesteps
                    current_action = actions[0].reshape(1, -1)
                    action = actions[t:t+20]
                    action_padded = np.zeros((20, action.shape[1]))
                    action_padded[:action.shape[0], :] = action
                    batch_actions.append(action_padded)
                    batch_states.append(state.reshape(1, -1))

                    if t!=0:
                        batch_next_states.append(state.reshape(1, -1))

                    if t == T - 1:
                        next_obs, reward, terminated, truncated, info = env.step(current_action)
                        next_state = next_obs['agent']['qpos']
                        batch_next_states.append(next_state.reshape(1, -1))

                        for act, st, img_emb, next_st in zip(
                            batch_actions, batch_states, batch_images, batch_next_states
                        ):
                            yield {
                                'action': act,
                                'state': st,
                                'image_emb': img_emb,
                                'next_state': next_st
                            }

                        # Reset batch
                        batch_images, batch_actions, batch_states, batch_next_states = [], [], [], []

    dataset = datasets.Dataset.from_generator(
            generator,
            cache_dir=config.CACHE_DIR,
        )
    dataset.save_to_disk(config.DEMONSTRATIONS_DIR_EE_PUSH_CUBE_RESNET_IMAGE)
    dataset.cleanup_cache_files()

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(device)

    args = argparse.ArgumentParser()
    args.add_argument('--env_id', type=str, default="PushCube-v1", help='Environment ID')
    args.add_argument('--control_mode', type=str, default="pd_ee_delta_pos", help='Control mode for the environment')
    args.add_argument('--checkpoint', type=str,
                       default="/gayu-west/act_full_network/runs/act-PushCube-v1-state-100_motionplanning_demos-1/checkpoints/best_eval_success_at_end.pt",
                       help='ACT checkpoint to load the trained resnet backbone from')
    args = args.parse_args()

    #ACT's image backbone: resnet18 with frozen BatchNorm, layer4 output, loaded from a trained ACT checkpoint
    model_img = Backbone("resnet18", train_backbone=False, return_interm_layers=False, dilation=False)
    model_img = load_trained_backbone(model_img, args.checkpoint, device).to(device).eval()
    processor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #push cube path
    traj_path = "/gayu-west/robot-manipulation/PushCube-v1/trajectory.state.pd_ee_delta_pos.cpu.h5"
    process_data_from_h5(
            model_img=model_img,
            processor=processor,
            device=device,
            traj_path=traj_path,
            env_id=args.env_id,
            control_mode=args.control_mode
        )
