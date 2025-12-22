'''
this is robomimic_lowdim_runner_classifier_total.py

'''
import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import math
import wandb.sdk.data_types.video as wv
from SI_policy.gym_util.async_vector_env import AsyncVectorEnv
from SI_policy.gym_util.sync_vector_env import SyncVectorEnv
from SI_policy.gym_util.multistep_wrapper import MultiStepWrapper
from SI_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from SI_policy.model.common.rotation_transformer import RotationTransformer
import cv2
from SI_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from SI_policy.common.pytorch_util import dict_apply
from SI_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from SI_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import time

import requests
import numpy as np
from PIL import Image
import io

# Assuming 'images' is your numpy array
def convert_numpy_to_pil(arr):
    """Convert numpy array to PIL Image"""
    # Make sure array is in the right format for PIL
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(arr)
    return pil_image
#%%
def ask_vlm(images,dict,url):
    # Convert numpy array to PIL Image
    pil_image = convert_numpy_to_pil(images)

    # Save the PIL image to a bytes buffer
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)  # Reset pointer to beginning of buffer

    prompt = '''
    simply pick the label from the sample images you see in prompt and tell me which one is closest to what you see
    '''

    url = f"http://{url}:8236/predict"

    # Send as multipart form data
    files = {
        "image_file": ("image.png", img_buffer, "image/png"),
    }
    data = {
        "prompt": prompt
    }

    response = requests.post(url, files=files, data=data)
    result = response.json()
    # print(result)
    return result
def create_env(env_meta, obs_keys):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys,
        'rgb': ['agentview_image']})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=True,
        use_image_obs=True, 
    )
    return env

import joblib
#%%
from datetime import datetime


class RobomimicLowdimRunner(BaseLowdimRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            obs_keys,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            n_latency_steps=0,
            render_hw=(256,256),
            render_camera_name='agentview',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        """
        Assuming:
        n_obs_steps=2
        n_latency_steps=3
        n_action_steps=4
        o: obs
        i: inference
        a: action
        Batch t:
        |o|o| | | | | | |
        | |i|i|i| | | | |
        | | | | |a|a|a|a|
        Batch t+1
        | | | | |o|o| | | | | | |
        | | | | | |i|i|i| | | | |
        | | | | | | | | |a|a|a|a|
        """

        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        
        # steps_per_render = max(robosuite_fps // fps, 1)
        steps_per_render=1 # ryan added

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys
                )
            # hard reset doesn't influence lowdim env
            # robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                    VideoRecordingWrapper(
                        RobomimicLowdimWrapper(
                            env=robomimic_env,
                            obs_keys=obs_keys,
                            init_state=None,
                            render_hw=render_hw,
                            render_camera_name=render_camera_name
                        ),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=crf,
                            thread_type='FRAME',
                            thread_count=1
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render
                    ),
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps
                )
        
        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicLowdimWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        print("test_start_seed: ",test_start_seed)
        for i in range(n_test):
            print("i: ",i)
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicLowdimWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        print('here 1')
        # env = AsyncVectorEnv(env_fns)
        env = SyncVectorEnv(env_fns)
        print('here 2')
        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseLowdimPolicy,label_classifier,classifier_bool,TS,timestamp1,seed_number,folder_name,env_original,url,classifier_type):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_obs_paths= [None] * n_inits
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_chunks_obs=[]

        # env_name='Lift'
        env_name = self.env_meta['env_name']
        folder=folder_name
        #make dir
        os.makedirs(folder, exist_ok=True)


        for chunk_idx in range(n_chunks):
            obs_list=[]
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"rollout robomimic lowdim runner {env_name}Lowdim {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            # print("Observation type:", type(obs))
            # print("Observation shape/structure:", obs.shape if hasattr(obs, 'shape') else obs)

            index2=0
            done = False
            start_time=time.time()
            vlm_time_total=0
            cnn_time_total=0
            SIP_time_total=0
            sim_time_total=0
            process_time_total=0
            while not done:
                done_start=time.time()
                # lowdim_parts = []
                # lowdim_parts.append(obs['object'])
                # lowdim_parts.append(obs['robot0_eef_pos'])
                # lowdim_parts.append(obs['robot0_eef_quat'])
                # lowdim_parts.append(obs['robot0_gripper_qpos'])

                # Concatenate all parts along the last dimension
                # lowdim_obs = np.concatenate(lowdim_parts, axis=-1)
                lowdim_obs=obs
                # # Get image observation for classifier if needed
                # # image_obs = obs['agentview_image']
                # images = env.call('render', mode='rgb_array')
                # print type of images
                video_wrapper = env.envs[0].env  # VideoRecordingWrapper
                images=video_wrapper.env.render(mode='rgb_array')
                os.makedirs('data1/env_runner_image', exist_ok=True)
                local_image_path = "data1/env_runner_image/example.png"
                cv2.imwrite(local_image_path, images)
                vlm_time=0
                cnn_time=0
                dict1={
                    "i":2,
                    "n":2,
                    "t":10,
                    "s":50,
                    "c":100,
                    "g":2
                }
                if classifier_type=='vlm':
                    print('using vlm classifier')
                    start2=time.time()
                    output=ask_vlm(images,dict1,url)
                    end2=time.time()
                    vlm_time=end2-start2
                    try:
                        chosen_ts=dict1[output['prediction']]
                    except:
                        chosen_ts=2
                        print("error in chosen_ts: ",chosen_ts)
                    output['TS']=chosen_ts
                    label=output['prediction']
                    print("time taken to ask vlm: ",end2-start2,label)
                    seed1=self.env_seeds
                    seed1=seed1[0]
                elif classifier_type=='cnn':
                    print('using cnn classifier')
                    X_test_np=obs[:,1,:]
                    start3=time.time()
                    label_hat = label_classifier.predict(X_test_np)
                    end3=time.time()
                    cnn_time=end3-start3
                    label_hat=int(label_hat)
                    label=label_hat
                    for index,ts in enumerate(policy.TS_list):
                        if label_hat==int(index):
                            policy.TS=int(ts)
                            break
                    chosen_ts=policy.TS
                    print(label_hat,'policy.TS:',policy.TS)
                    seed1=self.env_seeds
                    seed1=seed1[0]
                else:
                    print('using none of vlm or cnn')
                    seed1=self.env_seeds
                    seed1=seed1[0]
                    chosen_ts=TS
                    label=TS
                # path join folder, seed1
                folder2=os.path.join(folder,str(seed1))
                # make dir
                os.makedirs(folder2, exist_ok=True)
                filename=f"index{index2}_{label}_TS{chosen_ts}.png"
                cv2.imwrite(os.path.join(folder2,filename), images)

                policy.TS=chosen_ts
                
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': lowdim_obs[:,:self.n_obs_steps].astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))




                X_test_np=lowdim_obs[:,1,:]


                # run policy
                with torch.no_grad():
                    start3=time.time()
                    action_dict = policy.predict_action(obs_dict)
                    end3=time.time()
                    SIP_time=end3-start3

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                start4=time.time()
                obs, reward, done, info = env.step(env_action)
                end4=time.time()
                sim_time=end4-start4
                done = np.all(done)

                X_test_np=obs[:,1,:]
                obs_list.append(X_test_np)

                past_action = action

                # update pbar
                pbar.update(action.shape[1])
                index2+=1
                vlm_time_total+=vlm_time
                cnn_time_total+=cnn_time
                SIP_time_total+=SIP_time
                sim_time_total+=sim_time
                done_end=time.time()
                done_taken=done_end-done_start
                process_time=done_taken-(vlm_time+SIP_time+sim_time+cnn_time)
                process_time_total+=process_time
            pbar.close()
            end_time=time.time()
            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            torch_obs = torch.from_numpy(np.stack(obs_list)    ) # more efficient conversion
            # all_obs_paths[this_global_slice] = torch_obs
            all_chunks_obs.append(torch_obs)

            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

            time_taken=end_time-start_time
        stacked_obs = torch.stack(all_chunks_obs, dim=0)
        # just write in folder/time_taken.txt
        
        result_dict={
        "time_taken":time_taken,
        'vlm_time_total':vlm_time_total,
        'cnn_time_total':cnn_time_total,
        'SIP_time_total':SIP_time_total,
        'sim_time_total':sim_time_total,
        'process_time_total':process_time_total,
        'done_taken':done_taken,
        'vlm_time':vlm_time,
        'cnn_time':cnn_time,
        'SIP_time':SIP_time,
        'sim_time':sim_time,
        'process_time':process_time,
        'iter_repeat':n_inits,
        'index2':index2,
        "seed":seed_number,
        "success":done,
        "reward":reward
        }
        print("result_dict: ",result_dict)
        # convert result_dict to string
        # result_dict_str=json.dumps(result_dict)
        # save by appending with seed number in folder/time_taken.txt
        with open(os.path.join(folder,f"time_taken.txt"),"a") as f:
            f.write(f"{result_dict}\n")
            # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            obs_stack= stacked_obs[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
                # ryan added below
                log_data[prefix+f'sim_video_{seed}_obs'] = obs_stack

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
