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
from diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler import DiffusionUnetLowdimPolicy_flow_euler
# from diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler import DiffusionUnetLowdimPolicy_flow_euler
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
        print("cfg.policy.TS: ",cfg.policy.TS)
        cfg.logging.name=f'{cfg.logging.name}_{cfg.training.architecture}_{cfg.policy.DDPM}_W{cfg.policy.WSCALE}_TS{cfg.policy.TS}_{cfg.policy.diffusion_form}'
        if output_dir is None:
            output_dir = os.path.join('data', 'output2', scriptname, f'{default_timestamp}_{cfg.logging.name}')
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
        # self.model.T=cfg.training.T
        # self.model.TS=cfg.training.TS
        # self.model.WSCALE=cfg.training.WSCALE
        # self.model.DDPM=cfg.training.DDPM

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
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
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

        load_dict={}
        
        data2 = os.listdir(cfg.training.save_path)
        data2=sorted(data2)
        for item in data2:
            fullpath=os.path.join(cfg.training.save_path,item)
            if os.path.isfile(fullpath):
                if '.ckpt' in item and '=' in item:
                    # data/output2/a15_6_unet_linear_abs_DDPM/20240923_084752_lift_lowdim_abs_unet_DDPM_W100_TS2/checkpoints/epoch=0000-test_mean_score=0.020.ckpt
                    
                    epoch_num=int(item.split('=')[1].split('-')[0])
                    load_dict[epoch_num]=fullpath
                    

        # input(f'{load_dict},,, \n next?')


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
        # topk_manager = TopKCheckpointManager(
        #     save_dir=os.path.join(self.output_dir, 'checkpoints'),
        #     **cfg.checkpoint.topk
        # )

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
        SAVE_EPOCH=50
        # log_path = os.path.join(self.output_dir, 'logs.json.txt')

        # g.training.save_path='data/output2/a15_6_unet_linear_abs_DDPM/20240923_084752_lift_lowdim_abs_unet_DDPM_W100_TS2/checkpoints'
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            start1=time.time()
            # ========= train for this epoch ==========
            if local_epoch_idx % SAVE_EPOCH==0 and local_epoch_idx>3499:
                try:
                    ckpt_path=load_dict[local_epoch_idx]
                    model_name=ckpt_path.split('/')[-1]
                    print(f"Resuming from checkpoint {ckpt_path}")
                    print("device: ",device)
                    self.load_checkpoint(path=ckpt_path)
                except Exception as e:
                    print('success done- ',e)
                    break
            else:
                wandb_run.log(step_log, step=local_epoch_idx)
                continue

            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    # device transfer
                    break
            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model

            self.model.to(device)
            if self.ema_model is not None:
                self.ema_model.to(device)
            optimizer_to(self.optimizer, device)

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
                # print("mean_score_train: ",mean_score_train)
                # print("mean_score_test: ",mean_score_test)
                step_log['test_mean_score'] = mean_score_test
                step_log['train_mean_score'] = mean_score_train
                average_test_success.update(mean_score_test)
                average_train_success.update(mean_score_train)

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

            # run diffusion sampling on a training batch
            # if (self.epoch % cfg.training.rollout_every) == 0:
            #     with torch.no_grad():
            #         # sample trajectory from training set, and evaluate difference
            #         batch = train_sampling_batch
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
                    
                    # grid=0
                    # if grid==0:
                    #     global epoch
                    #     script_name=os.path.basename(__file__).replace('.py','')
                    #     # script_name = remove_prefix(script_name)
                    #     image_mse=mse.item()
                    #     # agent_pos=unnormalize_data(agent_pos,stats['agent_pos'])
                    #     # unnormalized_agent_pos=batch['obs']['agent_pos']
                    #     # unnormalized_agent_pos=torch.randint(low=1,high=2,size=(B,Da)).to(device)
                    #     unnormalized_agent_pos=batch['obs']['agent_pos']
                    #     # img=batch['img']
                    #     unnormalized_img=batch['obs']['image']
                        

                    #     unnormalized_a_true=gt_action
                    #     unnormalized_a_hat=pred_action
                    #     epoch=local_epoch_idx
                    #     # global model_name
                    #     title=f'{cfg.logging.name}_{model_name}'
                    #     save_grid_v4(title,unnormalized_img,unnormalized_a_true,unnormalized_a_hat,unnormalized_agent_pos,epoch,image_mse)
                    # release RAM
                    # del batch
                    # del obs_dict
                    # del gt_action
                    # del result
                    # del pred_action
                    # del mse
            
            # end of epoch
            # log of last step is combined with validation and rollout
            step_log['test_mean_score_last10'] = average_test_success.value
            step_log['train_mean_score_last10'] = average_train_success.value
            wandb_run.log(step_log, step=local_epoch_idx)
            # json_logger.log(step_log)
            self.global_step += 1
            self.epoch += 1
            end1=time.time()
            time_took=end1-start1
            time_took=round(time_took,2)
            if local_epoch_idx%1==0:
                print('logging.name',cfg.logging.name,'scriptname:',scriptname,'logging.project:',cfg.logging.project)
                print('processes: ',print_process_counts())
                print('train_success:',mean_score_train,'test_success:',mean_score_test)
                print('avg10_train_success:',average_train_success.value,'avg10_test_success:',average_test_success.value)

                # print('to complete 1000 epochs:',int(time_took*1000/60/60),'hours')
                print(' ')
            print(f"epoch {local_epoch_idx} - {int(time_took)} sec",'eval 20 epochs: ',int(time_took*20/60/60),'hours')
            

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
+cfg.training.save_path='data/output2/a15_6_unet_linear_abs_DDPM/20240923_084752_lift_lowdim_abs_unet_DDPM_W100_TS2/checkpoints'


########################### 1 lift

python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=lift_lowdim_abs \
    logging.name=lift_lowdim_abs \
    logging.project=unet_lift_linear_new_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=SDE_CURVE \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083252_lift_lowdim_abs_unet_SDE_CURVE_W1000_TS2/checkpoints \
    +policy.WSCALE=1000 \
    +policy.TS=1

########################### 2 tool_hang

python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_linear_new_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=SDE_CURVE \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083622_tool_hang_lowdim_abs_unet_SDE_CURVE_W1000_TS2/checkpoints \
    +policy.WSCALE=1000 \
    +policy.TS=2



########################### 3 transport

python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_linear_new_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=SDE_CURVE \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083626_transport_lowdim_abs_unet_SDE_CURVE_W1000_TS2/checkpoints \
    +policy.WSCALE=1000 \
    +policy.TS=3




########################### 4 square


python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_new_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100415_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=ODE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.TS=2





########################### 5 can


python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=can_lowdim_abs  \
    logging.name=can_lowdim_abs \
    logging.project=unet_can_linear_new_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=SDE_CURVE \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083634_can_lowdim_abs_unet_SDE_CURVE_W1000_TS2/checkpoints \
    +policy.WSCALE=1000 \
    +policy.TS=1


########## 6 block push


python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_seed_abs  \
    logging.name=blockpush_lowdim_seed_abs \
    logging.project=unet_blockpush_linear_new_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=SDE_CURVE \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083932_blockpush_lowdim_seed_abs_unet_SDE_CURVE_W1000_TS2/checkpoints \
    +policy.WSCALE=1000 \
    +policy.TS=1



########## 7 kitchen

export MUJOCO_GL=egl 
python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_linear_new_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=SDE_CURVE \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_085046_kitchen_lowdim_abs_unet_SDE_CURVE_W1000_TS2/checkpoints \
    +policy.WSCALE=1000 \
    +policy.TS=1






'''