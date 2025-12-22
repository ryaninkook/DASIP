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

def save_video(runner_log,local_epoch_idx,output_dir_media):
    # global local_epoch_idx
    # Loop through the videos and rename them based on the rewards
    for i in range(6):
        # Handle different keys for train and test videos
        if i < 2:
            video_key = f'train/sim_video_{i}'
            reward_key = f'train/sim_max_reward_{i}'
        else:
            video_key = f'test/sim_video_10000{i-2}'
            reward_key = f'test/sim_max_reward_10000{i-2}'
        try:
            # Get the max reward and round it
            max_reward = runner_log[reward_key]
            max_reward = round(max_reward, 2)

            # Get the original video path
            video_path = runner_log[video_key]._path

            # Create the new video name with the reward
            new1 = video_path.replace('.mp4', '') + f'_{max_reward}.mp4'
            new_name = new1.split('/')[-1]
            new_name=f'{local_epoch_idx}_{new_name}'
            # Move the video to the new directory
            shutil.move(video_path, os.path.join(output_dir_media, new_name))

            # Print the new path for debugging
            # print(f"Moved video to: {os.path.join(output_dir_media, new_name)}")
        except Exception as e:
            print('error save video,',i,e)
    # Save the runner_log to a JSON file in the new directory
    log_path = os.path.join(output_dir_media, 'log',f'{local_epoch_idx}_log.json')
    with open(log_path, 'w') as json_file:
        json.dump(runner_log, json_file, default=str)  # default=str to handle non-serializable objects like WandB objects

# %%
class TrainDiffusionUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        cfg.logging.name=f'{cfg.logging.name}_{cfg.training.architecture}_{cfg.training.DDPM}_W{cfg.training.WSCALE}_TS{cfg.training.TS}'
        
        # if output_dir is None:
        output_dir = os.path.join('data', 'output2', scriptname, f'{default_timestamp}_{cfg.logging.name}')
        if cfg.training.resume:
            output_dir=cfg.training.save_path
            print("save path called! output_dir: ",output_dir)
            time.sleep(3)
        os.makedirs(output_dir, exist_ok=True)

        # fol1=f'data/output2/{scriptname}/media'
        # output_dir
        output_dir_media = os.path.join(output_dir,'media')
        os.makedirs(output_dir_media, exist_ok=True)
        os.makedirs(os.path.join(output_dir_media,'log'), exist_ok=True)
        self.output_dir_media=output_dir_media

        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetLowdimPolicy_flow_euler
        self.model = hydra.utils.instantiate(cfg.policy)
        self.model.T=cfg.training.T
        self.model.TS=cfg.training.TS
        self.model.WSCALE=cfg.training.WSCALE
        self.model.DDPM=cfg.training.DDPM

        self.ema_model: DiffusionUnetLowdimPolicy_flow_euler = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        print("cfg.training.resume: ",cfg.training.resume)
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            print("lastest_ckpt_path: ",lastest_ckpt_path)
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
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

        # configure env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

        # configure logging
        print("cfg.task: ",cfg.task)
        print("cfg.task.name: ",cfg.task.name)
        print("cfg.logging.project: ",cfg.logging.project)
        print("cfg.logging.name: ",cfg.logging.name)
        
        print("cfg.logging.name: ",cfg.logging.name)
        # input('next?')
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
        average_test_success=Average_past(10)
        average_train_success=Average_past(10)
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        self.model.path_type=cfg.training.path_type
        print("device: ",device)
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                start1=time.time()
                if cfg.training.resume:
                    if local_epoch_idx<self.epoch:
                        continue
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            # wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    # step_log.update(runner_log)
                    save_video(runner_log,local_epoch_idx,self.output_dir_media)
                    
                    mean_score_test = runner_log['test/mean_score']
                    mean_score_train = runner_log['train/mean_score']
                    mean_score_test=round(mean_score_test,2)
                    mean_score_train=round(mean_score_train,2)
                    print("mean_score_train: ",mean_score_train)
                    print("mean_score_test: ",mean_score_test)
                    step_log['test_mean_score'] = mean_score_test
                    step_log['train_mean_score'] = mean_score_train
                    average_test_success.update(mean_score_test)
                    average_train_success.update(mean_score_train)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        if cfg.pred_action_steps_only:
                            pred_action = result['action']
                            start = cfg.n_obs_steps - 1
                            end = start + cfg.n_action_steps
                            gt_action = gt_action[:,start:end]
                        else:
                            pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        # log
                        step_log['train_action_mse_error'] = mse.item()
                        # release RAM
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing


                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    
                    if cfg.training.save_50 and self.epoch > int(cfg.training.save_from):
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict,True)
                        self.save_checkpoint(path=topk_ckpt_path)
                    else:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                step_log['test_mean_score_last10'] = average_test_success.value
                step_log['train_mean_score_last10'] = average_train_success.value
                wandb_run.log(step_log, step=local_epoch_idx)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
                end1=time.time()
                time_took=end1-start1
                time_took=round(time_took,2)
                if local_epoch_idx%10==0:
                    print('logging.name',cfg.logging.name,'scriptname:',scriptname,'logging.project:',cfg.logging.project)
                    print('processes: ',print_process_counts())
                    print('train_success:',mean_score_train,'test_success:',mean_score_test)
                    print('avg10_train_success:',average_train_success.value,'avg10_test_success:',average_train_success.value)

                    # print('to complete 1000 epochs:',int(time_took*1000/60/60),'hours')
                    print(' ')
                print(f"epoch {local_epoch_idx} - {int(time_took)} sec",'1000 epochs: ',int(time_took*1000/60/60),'hours')
                


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

'''


########################### 1 lift

python diffusion_policy/run1/a15_3_unet_linear_interpol_mimic_abs.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=lift_lowdim_abs \
    logging.name=lift_lowdim_abs \
    logging.project=unet_lift_flow_linear_interpol_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.DDPM=SDE_CURVE \
    +training.WSCALE=1000 \
    +training.T=100 \
    +training.TS=2 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083252_lift_lowdim_abs_unet_SDE_CURVE_W1000_TS2 \
    +training.path_type=linear \
    training.resume=true

########################### 2 tool_hang


python diffusion_policy/run1/a15_3_unet_linear_interpol_mimic_abs.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_flow_linear_interpol_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.DDPM=SDE_CURVE \
    +training.WSCALE=1000 \
    +training.T=100 \
    +training.TS=2 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083622_tool_hang_lowdim_abs_unet_SDE_CURVE_W1000_TS2 \
    +training.path_type=linear \
    training.resume=true



########################### 3 transport

python diffusion_policy/run1/a15_3_unet_linear_interpol_mimic_abs.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_flow_linear_interpol_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.DDPM=SDE_CURVE \
    +training.WSCALE=1000 \
    +training.T=100 \
    +training.TS=2 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083626_transport_lowdim_abs_unet_SDE_CURVE_W1000_TS2 \
    +training.path_type=linear \
    training.resume=true




########################### 4 square


python diffusion_policy/run1/a15_3_unet_linear_interpol_mimic_abs.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_flow_linear_interpol_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.DDPM=SDE_CURVE \
    +training.WSCALE=1000 \
    +training.T=100 \
    +training.TS=2 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083630_square_lowdim_abs_unet_SDE_CURVE_W1000_TS2 \
    +training.path_type=linear \
    training.resume=true






########################### 5 can


python diffusion_policy/run1/a15_3_unet_linear_interpol_mimic_abs.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=can_lowdim_abs  \
    logging.name=can_lowdim_abs \
    logging.project=unet_can_flow_linear_interpol_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.DDPM=SDE_CURVE \
    +training.WSCALE=1000 \
    +training.T=100 \
    +training.TS=2 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083634_can_lowdim_abs_unet_SDE_CURVE_W1000_TS2 \
    +training.path_type=linear \
    training.resume=true



########## 6 block push


python diffusion_policy/run1/a15_3_unet_linear_interpol_mimic_abs.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_seed_abs  \
    logging.name=blockpush_lowdim_seed_abs \
    logging.project=unet_blockpush_flow_linear_interpol_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.DDPM=SDE_CURVE \
    +training.WSCALE=1000 \
    +training.T=100 \
    +training.TS=2 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083932_blockpush_lowdim_seed_abs_unet_SDE_CURVE_W1000_TS2 \
    +training.path_type=linear \
    training.resume=true




########## 7 kitchen

export MUJOCO_GL=egl 
python diffusion_policy/run1/a15_3_unet_linear_interpol_mimic_abs.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_flow_linear_interpol_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.DDPM=SDE_CURVE \
    +training.WSCALE=1000 \
    +training.T=100 \
    +training.TS=2 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_085046_kitchen_lowdim_abs_unet_SDE_CURVE_W1000_TS2 \
    +training.path_type=linear \
    training.resume=true







'''