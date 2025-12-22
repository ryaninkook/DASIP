if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import shutil

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
# from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy_flow_euler
from diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler import DiffusionUnetLowdimPolicy_flow_euler
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.policy.diffusion_unet_lowdim_policy_ddim import DiffusionUnetLowdimPolicy_ddim
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel


import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from import_true_basics import *

OmegaConf.register_new_resolver("eval", eval, replace=True)

scriptname = os.path.basename(__file__).replace('.py', '')
default_timestamp = timestamp()

def save_video(runner_log, local_epoch_idx, output_dir_media):
    # Loop over keys in runner_log that contain "sim_video"
    for key in runner_log:
        if "sim_video" in key:
            # Determine if this is a test or train video
            if key.startswith("test/sim_video"):
                prefix, suffix = key.split("test/sim_video_")
                reward_key = f"test/sim_max_reward_{suffix}"
                obs_key = f"test/sim_video_{suffix}_obs"
            # elif key.startswith("train/sim_video"):
            #     prefix, suffix = key.split("train/sim_video_")
            #     reward_key = f"train/sim_max_reward_{suffix}"
            #     obs_key = f"train/sim_video_{suffix}_obs"
            else:
                # Skip if key does not match expected patterns.
                continue

            try:
                # Get the max reward from the log and round it
                max_reward = runner_log[reward_key]
                max_reward = round(max_reward, 2)

                # Retrieve the video object and its path.
                video_obj = runner_log[key]
                video_path = video_obj._path

                # Retrieve the observation tensor
                obs_torch = runner_log[obs_key]

                # Create a new video name with the reward included
                new_name = f'v2_{local_epoch_idx}_{os.path.basename(video_path).replace(".mp4", f"_{max_reward}.mp4")}'
                new_video_path = os.path.join(output_dir_media, new_name)

                # Move the video file to the new directory
                shutil.move(video_path, new_video_path)
                
                # Save the observation tensor (using the same basename with .pt extension)
                obs_path = os.path.join(output_dir_media, new_name.replace(".mp4", ".pt"))
                torch.save(obs_torch, obs_path)
                print(f"Successfully saved video {key}: {video_path} -> {new_video_path}")
            except Exception as e:
                print("Error saving video for key:", key, "Error:", e, "Video path:", video_path)
            # Process only the first matching video, then exit the loop
            

    # Save the complete runner_log as JSON for later reference.
    log_dir = os.path.join(output_dir_media, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{local_epoch_idx}_log.json")
    with open(log_path, "w") as json_file:
        json.dump(runner_log, json_file, default=str)


import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

import os
import cv2
import numpy as np
import torch

def save_obs(video_path):
    print('start - save obs')
    pt_path=video_path.replace('.mp4', '.pt')
    # pt_path   ='/data/ryan/code/robo/diffusion_policy/data/output4_train/square_lowdim/v11_square_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M/media/20250129_105600__TS10_[10, 10, 10, 10, 10, 10, 10]_square_lowdim_abs_unet_Euler_GVP2_velocity_ODE_Teps0_Seps0_Ssamuniform_____TS10/media/3500_d6bnkcra_1.0.pt'
    #%%
    unique_name=video_path.split('/')[-1].split('.mp4')[0]
    unique_name

    #%%
    origin=video_path.split('media')[0]
    origin
    #%%
    origin1=origin.replace('output4_train','output4_obs')
    #%%
    specific=video_path.split('media')[1]
    #%%
    media_path=origin+'media'+specific
    media_path1=origin1+'media'+specific
    # media_path=os.path.join(origin, 'media',specific)
    media_path

    #%%
    # read video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    frames = frames[..., ::-1]

    # read pt
    pt = torch.load(pt_path)

    # %%
    frames.shape
    #%%
    pt
    #%%
    pt.shape
    # # %%
    # frames_per_obs = frames.shape[0] // pt.shape[0]  # e.g., 400 // 50 = 8

    # for i in range(pt.shape[0]):
    #     obs_i = pt[i] 
    #     frame_start = i * frames_per_obs
    #     frame_end   = (i + 1) * frames_per_obs
    #     step_frames = frames[frame_start:frame_end]  # shape: (8, 128, 128, 3)
        

    frames_per_obs = frames.shape[0] // pt.shape[0]  # e.g. 8
    # save_dir = "linked_frames_obs"
    save_dir=os.path.join(media_path, 'obs',unique_name)
    save_dir1=os.path.join(media_path1, 'obs',unique_name)
    #%%
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir1, exist_ok=True)
    for i in range(pt.shape[0]):

        # 1) The observation at step i
        obs_i = pt[i].numpy()  # or keep it as torch

        # 2) The slice of frames for step i
        start_idx = i * frames_per_obs
        end_idx   = (i + 1) * frames_per_obs
        step_frames = frames[start_idx:end_idx]  # shape e.g. (8, H, W, 3)

        # For example, pick the last frame in the chunk
        rep_frame = step_frames[-1]

        # 3) Save the representative frame
        # Convert from RGB to BGR if you want to use cv2.imwrite
        rep_frame_bgr = rep_frame[..., ::-1]  
        frame_path = os.path.join(save_dir, f"obs_{i:03d}.png")
        cv2.imwrite(frame_path, rep_frame_bgr)

        frame_path1 = os.path.join(save_dir1, f"obs_{i:03d}.png")
        cv2.imwrite(frame_path1, rep_frame_bgr)
        # Also save the low‐dim obs as a .npy or something
        np.save(os.path.join(save_dir, f"obs_{i:03d}.npy"), obs_i)
        np.save(os.path.join(save_dir1, f"obs_{i:03d}.npy"), obs_i)
    # %%

    print('end - save obs')


# %%
class TrainDiffusionUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        if cfg.training.training:
            if cfg.policy.path_type=='DDPM' or cfg.policy.path_type=='DDIM':
                cfg.logging.name=f'{cfg.logging.name}_{cfg.training.architecture}_{cfg.policy.DDPM}_{cfg.policy.path_type}_{cfg.policy.prediction}'
            else:
                cfg.logging.name=f'{cfg.logging.name}_{cfg.training.architecture}_{cfg.policy.DDPM}_{cfg.policy.path_type}_{cfg.policy.prediction}_Teps{cfg.policy.train_eps}_Tsam{cfg.policy.Tsam}_T{cfg.policy.T}'
        else:
            if cfg.policy.path_type=='DDPM' or cfg.policy.path_type=='DDIM':
                cfg.logging.name=f'{cfg.logging.name}_{cfg.training.architecture}_{cfg.policy.DDPM}_{cfg.policy.path_type}_{cfg.policy.prediction}_{cfg.policy.ODE_SDE}_TS{cfg.policy.TS}'
            else:
                if cfg.policy.ODE_SDE=='ODE':
                    cfg.logging.name=f'_TS{cfg.policy.TS}_{cfg.policy.TS_list}_{cfg.logging.name}_{cfg.training.architecture}_{cfg.policy.DDPM}_{cfg.policy.path_type}_{cfg.policy.prediction}_{cfg.policy.ODE_SDE}_Teps{cfg.policy.train_eps}_Seps{cfg.policy.sample_eps}_Ssam{cfg.policy.Ssam}_{cfg.policy.beta_params}_{cfg.policy.beta_scaling}_TS{cfg.policy.TS}_{cfg.training.infer_from}'
                elif cfg.policy.ODE_SDE=='SDE':
                    cfg.logging.name=f'{cfg.logging.name}_{cfg.policy.TS_list}_{cfg.training.architecture}_{cfg.policy.DDPM}_{cfg.policy.path_type}_{cfg.policy.prediction}_{cfg.policy.ODE_SDE}_Teps{cfg.policy.train_eps}_Seps{cfg.policy.sample_eps}_{cfg.policy.diffusion_form}_{cfg.policy.last_step}_{cfg.policy.last_step_size}_Ssam{cfg.policy.Ssam}_{cfg.policy.beta_params}_{cfg.policy.beta_scaling}_TS{cfg.policy.TS}_{cfg.training.infer_from}'
        # if output_dir is None:
        # if cfg.training.training:
        
        if cfg.training.synology_true: # synology
            cfg.training.save_path=cfg.training.synology_path.split('/')[-1]
            self.checkpoint_dir=os.path.join(cfg.training.synology_path,'checkpoints')
            output_dir = os.path.join('data', 'output4_train', cfg.task.name , cfg.training.save_path)
        else:
            output_dir = os.path.join('data', 'output4_train', cfg.task.name , cfg.training.save_path)

            self.checkpoint_dir=os.path.join(output_dir,'checkpoints')

        if len(str(cfg.training.save_path))<5:
            raise TypeError('wrong save path')
        
        os.makedirs(output_dir, exist_ok=True) # synology
        video_dir=copy.deepcopy(output_dir)
        if not cfg.training.training:
            # self.checkpoint_dir=os.path.join(output_dir,'checkpoints')
            timedelta1=0
            timedelta2=100
            today1=((datetime.now(timezone('US/Eastern'))-timedelta(timedelta1)).strftime("%Y%m%d_%H%M%S"))
            today2=((datetime.now(timezone('US/Eastern'))-timedelta(timedelta1))).timestamp()
            past1=((datetime.now(timezone('US/Eastern'))-timedelta(timedelta2)).strftime("%Y-%m-%d"))
            past2=datetime.strptime('20210218', "%Y%m%d").timestamp()
            output_dir=os.path.join(output_dir,'media',f'{today1}_{cfg.logging.name}')
        print("save path called! output_dir: ",output_dir)
        time.sleep(3)

        # else:
        # output_dir = os.path.join('data', 'output4_infer',scriptname, cfg.task.name, f'{default_timestamp}_{cfg.logging.name}')
        os.makedirs(output_dir, exist_ok=True)

        # fol1=f'data/output2/{scriptname}/media'
        # output_dir
        # self.output_dir=output_dir
        # if cfg.training.training:
        output_dir_media = os.path.join(output_dir,'media')
        os.makedirs(output_dir_media, exist_ok=True)
        os.makedirs(os.path.join(output_dir_media,'log'), exist_ok=True)

        # output_dir_media = os.path.join(output_dir,'media',f'{default_timestamp}_{cfg.logging.name}')
        # os.makedirs(output_dir_media, exist_ok=True)
        
        os.makedirs(os.path.join(output_dir_media,'log'), exist_ok=True)
        self.output_dir_media=output_dir_media

        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model

        # configure model
        if cfg.policy.path_type=='DDPM':
            self.model: DiffusionUnetLowdimPolicy
            self.ema_model: DiffusionUnetLowdimPolicy = None
        elif cfg.policy.path_type=='DDIM':
            self.model: DiffusionUnetLowdimPolicy_ddim
                
            OmegaConf.set_struct(cfg.policy.noise_scheduler, False)

            if 'variance_type' in cfg.policy.noise_scheduler:
                cfg.policy.noise_scheduler.pop('variance_type')

            # Re-enable struct mode
            OmegaConf.set_struct(cfg.policy.noise_scheduler, True)

            cfg.policy.noise_scheduler._target_= 'diffusers.schedulers.scheduling_ddim.DDIMScheduler'
            cfg.policy.noise_scheduler.num_train_timesteps=100
            cfg.policy.noise_scheduler.beta_start=0.0001
            cfg.policy.noise_scheduler.beta_end=0.02
            cfg.policy.noise_scheduler.beta_schedule='squaredcos_cap_v2'
            cfg.policy.noise_scheduler.clip_sample=True

            self.model = hydra.utils.instantiate(cfg.policy)
            print("self.model.noise_scheduler: ",self.model.noise_scheduler)
            self.model.num_inference_steps=cfg.policy.TS


            self.ema_model: DiffusionUnetLowdimPolicy_ddim = None

        else:
            self.model: DiffusionUnetLowdimPolicy_flow_euler
            self.ema_model: DiffusionUnetLowdimPolicy_flow_euler = None


        self.model = hydra.utils.instantiate(cfg.policy)
        
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

        import joblib
        
        task_name=cfg.task.name.split('_')[0]
        # if task_name=='square':
        #     label_path='/data/ryan/code/robo/diffusion_policy/data/flow_label/square_xg_93_c1_learning_rate0.8_max_depth9_n_estimators100_1.pkl'
        # elif ....

        if task_name == 'square':
            label_path = 'data/flow_label/square_xg_93_c1_learning_rate0.8_max_depth9_n_estimators100_1.pkl'
            label_path='data/flow_label/v2/square_xg_97_c0_learning_rate0.1_max_depth3_n_estimators50_0.pkl'
        elif task_name == 'can':
            label_path = 'data/flow_label/can_xg_98_c1_learning_rate0.1_max_depth3_n_estimators200_1.pkl'
            label_path='data/flow_label/v2/can_xg_99_c0_learning_rate0.1_max_depth3_n_estimators50_0.pkl'
        elif task_name == 'lift':
            label_path = 'data/flow_label/lift_rf_95_c1_max_depth9_n_estimators50_1.pkl'
            label_path='data/flow_label/v2/lift_xg_99_c0_learning_rate0.1_max_depth3_n_estimators50_0.pkl'
        elif task_name == 'pusht':
            label_path = 'data/flow_label/pusht_xg_87_c2_learning_rate0.4_max_depth12_n_estimators100_2.pkl'
            label_path = 'data/flow_label/v2/pusht_xg_87_c0_learning_rate0.1_max_depth3_n_estimators50_0.pkl'
        elif task_name == 'tool':
            label_path = 'data/flow_label/tool_hang_rf_94_c2_max_depth15_n_estimators200_2.pkl'
            label_path='data/flow_label/v2/tool_hang_xg_89_c1_learning_rate0.1_max_depth3_n_estimators50_1.pkl'
        elif task_name == 'transport':
            label_path = 'data/flow_label/transport_xg_95_c1_learning_rate0.6_max_depth9_n_estimators50_1.pkl'
            label_path='data/flow_label/v2/transport_xg_94_c1_learning_rate0.1_max_depth3_n_estimators50_1.pkl'
        else:
            raise ValueError(f"Unknown task_name: {task_name}")
        
        self.label_classifier = joblib.load(label_path)
        


    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                # random from 0 to 20 sec sleep below
                print('loading success')
                time.sleep(random.randint(0,20))

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
        print("self.output_dir: ",self.output_dir)
        time.sleep(3)
        # configure env runner
        print("cfg.task.env_runner.test_start_seed: ",cfg.task.env_runner.test_start_seed)
        cfg.task.env_runner.test_start_seed=cfg.training.seed
        print("cfg.task.env_runner.test_start_seed: ",cfg.task.env_runner.test_start_seed)
        if 'push' in cfg.task.name:
            cfg.task.env_runner._target_='diffusion_policy.env_runner.pusht_keypoints_runner_classifier.PushTKeypointsRunner'
        else:
            cfg.task.env_runner._target_='diffusion_policy.env_runner.robomimic_lowdim_runner_classifier.RobomimicLowdimRunner'


        # configure logging
        print("cfg.task: ",cfg.task)
        print("cfg.task.name: ",cfg.task.name)
        print("cfg.logging.project: ",cfg.logging.project)
        print("cfg.logging.name: ",cfg.logging.name)
        
        print("cfg.logging.name: ",cfg.logging.name)
        # input('next?')
# 

        load_dict={}
        if not cfg.training.training:
            data2 = os.listdir(self.checkpoint_dir)
            data2=sorted(data2)
            print("self.checkpoint_dir: ",self.checkpoint_dir)
            print("checkpoint_dir os listdir: ",data2)
            for item in data2:
                fullpath=os.path.join(self.checkpoint_dir,item)
                if os.path.isfile(fullpath):
                    if '.ckpt' in item and '=' in item:
                        # data/output2/a15_6_unet_linear_abs_DDPM/20240923_084752_lift_lowdim_abs_unet_DDPM_W100_TS2/checkpoints/epoch=0000-test_mean_score=0.020.ckpt
                        
                        epoch_num=int(item.split('=')[1].split('-')[0])
                        load_dict[epoch_num]=fullpath
                        

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        # device = torch.device(cfg.training.device)
        device=find_empty_gpu_v2()
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
        mean_score_test=0
        mean_score_train=0
        average_test_success=Average_past(50000,True)
        average_train_success=Average_past(50000,True)
        average_test_time_took=Average_past(50000,True)

        # training loop
        print("device: ",device)
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        
        ckpt_path=load_dict[int(cfg.training.infer_from)]
        model_name=ckpt_path.split('/')[-1]
        print(f"Resuming from checkpoint {ckpt_path}")
        print("device: ",device)
        self.load_checkpoint(path=ckpt_path)
        
        
        with JsonLogger(log_path) as json_logger:
            for seed_number in range(cfg.training.num_epochs):
                
                cfg.task.env_runner.n_envs=1
                cfg.task.env_runner.n_train=0
                cfg.task.env_runner.n_test=1
                
                env_runner: BaseLowdimRunner
                env_runner = hydra.utils.instantiate(
                    cfg.task.env_runner,
                    output_dir=self.output_dir)
                assert isinstance(env_runner, BaseLowdimRunner)
                cfg.task.env_runner.test_start_seed+=cfg.task.env_runner.n_test
                print("cfg.task.env_runner.test_start_seed: ",cfg.task.env_runner.test_start_seed)
                
                
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                start1=time.time()
   

                if not cfg.training.training:

                    # ========= eval for this epoch ==========
                    policy = self.model
                    if cfg.training.use_ema:
                        policy = self.ema_model
                    self.model.to(device)
                    if self.ema_model is not None:
                        self.ema_model.to(device)

                    policy.eval()

                    # run rollout
                    # if (self.epoch % cfg.training.rollout_every) == 0:
                    if True:
                        classifier_bool=True
                        TS=cfg.policy.TS
                        env_start_time=time.time()
                        if 'pusht' in cfg.task.name:

                            runner_log = env_runner.run(policy,self.label_classifier,classifier_bool,TS,cfg.training.pusht_heun)
                        else:
                            runner_log = env_runner.run(policy,self.label_classifier,classifier_bool,TS)
                        # log all
                        env_end_time=time.time()

                        # step_log.update(runner_log)
                        save_video(runner_log,seed_number,self.output_dir_media)
                        
                        mean_score_test = runner_log['test/mean_score']
                        # mean_score_train = runner_log['train/mean_score']
                        mean_score_train=0
                        # mean_score_test=round(mean_score_test,2)
                        mean_score_train=round(mean_score_train,2)
                        env_time_took=round(env_end_time-env_start_time,2)  
                        # i want to save all the values in runner_log into step_log. how?
                        # step_log.update(runner_log)
                        # print("mean_score_train: ",mean_score_train)
                        # print("mean_score_test: ",mean_score_test)
                        step_log['test_mean_score'] = mean_score_test
                        step_log['train_mean_score'] = mean_score_train
                        step_log['env_time_took'] = env_time_took

                        average_test_success.update(mean_score_test)
                        average_train_success.update(mean_score_train)
                        average_test_time_took.update(env_time_took)

                    # # run validation
                    # if (self.epoch % cfg.training.val_every) == 0:
                    #     with torch.no_grad():
                    #         val_losses = list()
                    #         with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                    #                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    #             for batch_idx, batch in enumerate(tepoch):
                    #                 batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    #                 loss = self.model.compute_loss(batch)
                    #                 val_losses.append(loss)
                    #                 if (cfg.training.max_val_steps is not None) \
                    #                     and batch_idx >= (cfg.training.max_val_steps-1):
                    #                     break
                    #         if len(val_losses) > 0:
                    #             val_loss = torch.mean(torch.tensor(val_losses)).item()
                    #             # log epoch average validation loss
                    #             step_log['val_loss'] = val_loss

                    # # run diffusion sampling on a training batch
                    # if (self.epoch % cfg.training.sample_every) == 0:
                    #     with torch.no_grad():
                    #         # sample trajectory from training set, and evaluate difference
                    #         # batch = train_sampling_batch
                    #         obs_dict = {'obs': batch['obs']}
                    #         gt_action = batch['action']
                            
                    #         result = policy.predict_action(obs_dict)
                    #         if cfg.pred_action_steps_only:
                    #             pred_action = result['action']
                    #             start = cfg.n_obs_steps - 1
                    #             end = start + cfg.n_action_steps
                    #             gt_action = gt_action[:,start:end]
                    #         else:
                    #             pred_action = result['action_pred']
                    #         mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    #         # log
                    #         step_log['train_action_mse_error'] = mse.item()
                    #         # release RAM
                    #         del batch
                    #         del obs_dict
                    #         del gt_action
                    #         del result
                    #         del pred_action
                    #         del mse
                    

                # end of epoch
                # log of last step is combined with validation and rollout
                step_log['test_mean_score_last10'] = average_test_success.value
                step_log['train_mean_score_last10'] = average_train_success.value
                step_log['time_avg_took_to_complete'] = average_test_time_took.value

                wandb_run.log(step_log, step=seed_number)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
                end1=time.time()
                time_took=end1-start1
                time_took=round(time_took,2)
                if seed_number%10==0:
                    print('logging.name',cfg.logging.name,'scriptname:',scriptname,'logging.project:',cfg.logging.project)
                    print('processes: ',print_process_counts())
                    print('train_success:',mean_score_train,'test_success:',mean_score_test)
                    print('avg10_train_success:',average_train_success.value,'avg10_test_success:',average_test_success.value)
                    print("device: ",device)
                    # print('to complete 1000 epochs:',int(time_took*1000/60/60),'hours')
                    print(' ')
                print(f"epoch {seed_number} - {int(time_took)} sec",'1000 epochs: ',int(time_took*1000/60/60),'hours')
                


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
    # config_path='../config',
    # config_name='ryan_unet_lowdim_lift_lowdim_abs.yaml')
def main(cfg):


    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()


# %%
'''



python diffusion_policy/run4/a5_3_workspace.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=classifier_v2_test_infer_ode2_v11_square_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    +training.save_from=0.0 \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=v11_square_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M \
    training.resume=False \
    +policy.path_type=GVP2 \
    +policy.ODE_SDE=ODE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=. \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10 \
    +policy.last_step=. \
    +policy.last_step_size=. \
    +policy.train_eps=0 \
    +policy.sample_eps=0 \
    +policy.prediction=velocity \
    +policy.Tsam=uniform \
    +policy.Ssam=uniform \
    +policy.Tdiscrete=Tdiscrete \
    +policy.model.t_minus_one=t_minus_one \
    training.seed=2000000 \
    training.num_epochs=5000 \
    +policy.beta_params=_ \
    +policy.beta_scaling=_ \
    +training.training=False \
    +policy.TS_list=[10,10,10,10,10,10,10] \
    +training.infer_from=2000


main:[10,10,10,50,10,100,10]
all 100 : [100,100,100,100,100,100,100]

past : v11_transport_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M
now: v12_transport_train_Linear_velocity_uniform_Tconti_t_minus_one_seed1M

python diffusion_policy/run4/a5_3_workspace.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=classifier_v3_test_infer_ode2_v12_transport_train_Linear_velocity_uniform_Tconti_t_minus_one_seed1M \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    +training.save_from=0.0 \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=v12_transport_train_Linear_velocity_uniform_Tconti_t_minus_one_seed1M \
    training.resume=False \
    +policy.path_type=GVP2 \
    +policy.ODE_SDE=ODE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=. \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10 \
    +policy.last_step=. \
    +policy.last_step_size=. \
    +policy.train_eps=0 \
    +policy.sample_eps=0 \
    +policy.prediction=velocity \
    +policy.Tsam=uniform \
    +policy.Ssam=uniform \
    +policy.Tdiscrete=Tdiscrete \
    +policy.model.t_minus_one=t_minus_one \
    training.seed=2000000 \
    training.num_epochs=5000 \
    +policy.beta_params=_ \
    +policy.beta_scaling=_ \
    +training.training=False \
    +policy.TS_list=[2,2,100,2,2,2] \
    +training.infer_from=2000



python diffusion_policy/run4/a5_3_workspace.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=classifier_v2_test_infer_ode2_v11_tool_hang_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    +training.save_from=0.0 \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=v11_tool_hang_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M \
    training.resume=False \
    +policy.path_type=GVP2 \
    +policy.ODE_SDE=ODE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=. \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10 \
    +policy.last_step=. \
    +policy.last_step_size=. \
    +policy.train_eps=0 \
    +policy.sample_eps=0 \
    +policy.prediction=velocity \
    +policy.Tsam=uniform \
    +policy.Ssam=uniform \
    +policy.Tdiscrete=Tdiscrete \
    +policy.model.t_minus_one=t_minus_one \
    training.seed=2000000 \
    training.num_epochs=5000 \
    +policy.beta_params=_ \
    +policy.beta_scaling=_ \
    +training.training=False \
    +policy.TS_list=[10,10,10,10,10,10,10] \
    +training.infer_from=2000



python diffusion_policy/run4/a5_3_workspace.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=pusht_lowdim_abs  \
    logging.name=pusht_lowdim_abs \
    logging.project=classifier_v2_test_infer_ode2_v11_pusht_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    +training.save_from=0.0 \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=v11_pusht_train_vp_velocity_uniform_seed1 \
    training.resume=False \
    +policy.path_type=VP \
    +policy.ODE_SDE=ODE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=. \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10 \
    +policy.last_step=. \
    +policy.last_step_size=. \
    +policy.train_eps=0 \
    +policy.sample_eps=0 \
    +policy.prediction=velocity \
    +policy.Tsam=uniform \
    +policy.Ssam=uniform \
    +policy.Tdiscrete=Tdiscrete \
    +policy.model.t_minus_one=t_minus_one \
    training.seed=2000000 \
    training.num_epochs=5000 \
    +policy.beta_params=_ \
    +policy.beta_scaling=_ \
    +training.training=False \
    +policy.TS_list=[10,10,10,10,10,10,10] \
    +training.infer_from=500


python diffusion_policy/run4/a5_3_workspace.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=lift_lowdim_abs  \
    logging.name=lift_lowdim_abs \
    logging.project=classifier_v2_test_infer_ode2_v11_lift_train_gLinear2_velocity_uniform_Tdiscrete_t_minus_one_seed1M \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    +training.save_from=0.0 \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=v11_lift_train_Linear_velocity_uniform_Tconti_t_minus_one_seed1M \
    training.resume=False \
    +policy.path_type=Linear \
    +policy.ODE_SDE=ODE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=. \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10 \
    +policy.last_step=. \
    +policy.last_step_size=. \
    +policy.train_eps=0 \
    +policy.sample_eps=0 \
    +policy.prediction=velocity \
    +policy.Tsam=uniform \
    +policy.Ssam=uniform \
    +policy.Tdiscrete=Tconti \
    +policy.model.t_minus_one=t_minus_one \
    training.seed=2000000 \
    training.num_epochs=5000 \
    +policy.beta_params=_ \
    +policy.beta_scaling=_ \
    +training.training=False \
    +policy.TS_list=[10,10,10,10,10,10,10] \
    +training.infer_from=500




python diffusion_policy/run4/a5_3_workspace.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=can_lowdim_abs  \
    logging.name=can_lowdim_abs \
    logging.project=classifier_v2_test_infer_ode2_v11_can_train_gLinear2_velocity_uniform_Tdiscrete_t_minus_one_seed1M \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    +training.save_from=0.0 \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=v11_can_train_Linear_velocity_uniform_Tconti_t_minus_one_seed1M \
    training.resume=False \
    +policy.path_type=Linear \
    +policy.ODE_SDE=ODE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=. \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10 \
    +policy.last_step=. \
    +policy.last_step_size=. \
    +policy.train_eps=0 \
    +policy.sample_eps=0 \
    +policy.prediction=velocity \
    +policy.Tsam=uniform \
    +policy.Ssam=uniform \
    +policy.Tdiscrete=Tconti \
    +policy.model.t_minus_one=t_minus_one \
    training.seed=2000000 \
    training.num_epochs=5000 \
    +policy.beta_params=_ \
    +policy.beta_scaling=_ \
    +training.training=False \
    +policy.TS_list=[10,10,10,10,10,10,10] \
    +training.infer_from=500

'''


