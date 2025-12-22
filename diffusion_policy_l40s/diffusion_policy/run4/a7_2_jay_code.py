#%%
import os
import h5py
import numpy as np
# from flask import Flask, render_template, request, redirect, url_for
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper

# Initialize Flask app
# app = Flask(__name__)

TASK = 'pusht'

def create_env(env_meta, obs_keys):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=False,
        use_image_obs=False, 
    )
    return env

dataset_path = 'data/robomimic/datasets/{}/ph/low_dim_abs.hdf5'.format(TASK)
env_meta = FileUtils.get_env_metadata_from_dataset(
    dataset_path)

if TASK in ['tool_hang', 'can', 'lift', 'square']:
    obs_keys = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
elif TASK in ['transport']:
    obs_keys = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos']

robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys
                )
robomimic_env = RobomimicLowdimWrapper(
                        env=robomimic_env,
                        obs_keys=obs_keys,
                        init_state=None,
                        render_hw=(256,256),
                        render_camera_name='agentview'
                        )

#%%
# Variables to track state
episode_idx = 0
timestep_idx = 0
starting_episode = 0
total_episode = 500
labeled_data = []

# Generate and save all images and state observations for all episodes
# def generate_and_save_images():

os.makedirs('static/{}/images'.format(TASK), exist_ok=True)
os.makedirs('static/{}/data'.format(TASK), exist_ok=True)
with h5py.File(dataset_path, 'r') as f:
    for episode_idx in range(starting_episode, total_episode):  # Iterate over episodes 0 to 100
        episode_folder = f'static/{TASK}/images/episode_{episode_idx}'
        os.makedirs(episode_folder, exist_ok=True)
        episode_data = []
        max_timesteps = len(f[f'data/demo_{episode_idx}/states'])
        for t_idx in range(max_timesteps):
            init_state = f[f'data/demo_{episode_idx}/states'][t_idx]
            robomimic_env.init_state = init_state
            state_obs = robomimic_env.reset()
            img = robomimic_env.render()
            img_path = f'{episode_folder}/image_{t_idx}.png'
            from PIL import Image
            Image.fromarray(img).save(img_path)
            episode_data.append({"state_obs": state_obs, "img_path": img_path})
        np.save(f'static/{TASK}/data/episode_{episode_idx}.npy', episode_data)

# generate_and_save_images()