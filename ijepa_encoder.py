import datasets
from transformers import AutoProcessor, AutoModel
import gymnasium as gym
import config
import mani_skill.envs
import torch
import mani_skill.trajectory.utils as trajectory_utils
from tqdm import tqdm
import h5py
import numpy as np
import argparse

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

                    inputs = processor(images=img, return_tensors="pt").to(model_img.device)
                    with torch.inference_mode():
                        outputs = model_img(**inputs)

                    image_emb = outputs.last_hidden_state.mean(dim=1)
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
    dataset.save_to_disk(config.DEMONSTRATIONS_DIR_EE_PUSH_CUBE_JEPA_IMAGE)
    dataset.cleanup_cache_files()

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(device)

    model_id = "jmtzt/ijepa_vith16_1k"
    processor = AutoProcessor.from_pretrained(model_id)
    model_img = AutoModel.from_pretrained(model_id).to(device)

    args = argparse.ArgumentParser()
    args.add_argument('--env_id', type=str, default="PushCube-v1", help='Environment ID')
    args.add_argument('--control_mode', type=str, default="pd_ee_delta_pos", help='Control mode for the environment')
    args = args.parse_args()

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
