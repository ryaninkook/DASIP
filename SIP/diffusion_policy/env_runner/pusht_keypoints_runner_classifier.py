import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from gym.error import (
    AlreadyPendingCallError,
    NoAsyncCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
)

class PushTKeypointsRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            # n_test_vis=2,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        **kp_kwargs
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
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
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

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
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

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # env = AsyncVectorEnv(env_fns)
        env = SyncVectorEnv(env_fns)
        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    #         data2=env_runner.run(policy,label_classifier,classifier_bool,TS,timestamp1,seed_number,folder,env_original,url)

    def run(self, policy: BaseLowdimPolicy,label_classifier,classifier_bool,TS,timestamp1,seed_number,folder,env_original,url,euler_heun=0):
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

        for chunk_idx in range(n_chunks):
            obs_list=[]
            start = chunk_idx * n_envs
            
            end = min(n_inits, start + n_envs)
            print("chunk_idx: ",chunk_idx,start,end)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            # env.call_each('run_dill_function', 
                # args_list=[(x,) for x in this_init_fns])
            import time

            try:
                env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])
            except AlreadyPendingCallError:
                time.sleep(0.1)  # wait for a short time
                env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtKeypointsRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            start_time = time.time()
            while not done:
                Do = obs.shape[-1] // 2
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                if classifier_bool:

                    X_test_np=obs[:,1,:]
                    X_test_np=X_test_np[:,:20]
                    # print("\n>> X_test_np.shape= ", X_test_np.shape)

                    # obs_list.append(X_test_np)
                    # X_test_np  = obs

                    # 5) (Optional) Verify loaded model predictions
                    label_hat = label_classifier.predict(X_test_np)
                    label_hat=int(label_hat)
                    
                    for index,ts in enumerate(policy.TS_list):
                        if label_hat==int(index):
                            policy.TS=int(ts)
                            break
                    print(label_hat,'policy.TS:',policy.TS)
                else:
                    policy.TS=TS

                
                if euler_heun=='single_sde_heun_tweedie':
                    if label_hat==5:
                        policy.ODE_SDE='SDE'
                        policy.DDPM='Heun'
                        
                    else:
                        policy.ODE_SDE='ODE'
                        policy.DDPM='Euler'

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)

                X_test_np=obs[:,1,:]
                X_test_np=X_test_np[:,:20]

                obs_list.append(X_test_np)

                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()
            end_time = time.time()
            print("time: ",end_time-start_time)

            
            # collect data for this round
            # all_video_paths[this_global_slice] = env.render()[this_local_slice]
            # all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            path2=env.render()[this_local_slice]
            # print("path2: ",path2)
            rewards2=env.call('get_attr', 'reward')[this_local_slice]
            # print("rewar
            all_video_paths[this_global_slice] = path2
            # print("\n>> len(obs_list)= ", len(obs_list))

            torch_obs = torch.from_numpy(np.stack(obs_list)    ) # more efficient conversion
            # print("\n>> torch_obs.shape= ", torch_obs.shape)

            # all_obs_paths[this_global_slice] = torch_obs
            all_chunks_obs.append(torch_obs)

            all_rewards[this_global_slice] = rewards2
            max_reward2=max(rewards2)
            # print("max_reward2: ",max_reward2)
        # import pdb; pdb.set_trace()
        stacked_obs=all_chunks_obs
        # stacked_obs = torch.stack(all_chunks_obs, dim=0)

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
                log_data[prefix+f'sim_video_{seed}_obs'] = obs_stack

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
