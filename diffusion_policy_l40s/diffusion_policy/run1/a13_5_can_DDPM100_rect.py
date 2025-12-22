if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
import numba

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.robomimic_lowdim_policy import RobomimicLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to


import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from import_true_basics import *

OmegaConf.register_new_resolver("eval", eval, replace=True)

# below current script name without .py
scriptname = os.path.basename(__file__).replace('.py', '')

fol1=f'data/{scriptname}/media'
if not os.path.exists(fol1):
   os.makedirs(fol1)
#%%
# Directory where videos are currently located

data1 = os.listdir(fol1)
scriptname = os.path.basename(__file__).replace('.py', '')
default_timestamp = timestamp()
new_dir = os.path.join(fol1, f"{scriptname}",f"{default_timestamp}")
os.makedirs(new_dir, exist_ok=True)
os.makedirs(os.path.join(new_dir,'log'), exist_ok=True)

def save_video(runner_log,local_epoch_idx):
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
            shutil.move(video_path, os.path.join(new_dir, new_name))

            # Print the new path for debugging
            # print(f"Moved video to: {os.path.join(new_dir, new_name)}")
        except:
            pass
    # Save the runner_log to a JSON file in the new directory
    log_path = os.path.join(new_dir, 'log',f'{local_epoch_idx}_log.json')
    with open(log_path, 'w') as json_file:
        json.dump(runner_log, json_file, default=str)  # default=str to handle non-serializable objects like WandB objects

class TrainRobomimicLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: RobomimicLowdimPolicy = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self,wandb_name,wandb_project):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        # print("cfg.task.dataset: ",cfg.task.dataset)
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)

        # configure env
        # print("cfg.task.env_runner: ",cfg.task.env_runner)
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)
        cfg.logging.name=wandb_name
        cfg.logging.project=wandb_project
        # configure logging
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

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        mean_score_test=0
        mean_score_train=0
        average_test_success=Average_past(10)
        average_train_success=Average_past(10)
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        info = self.model.train_on_batch(batch, epoch=self.epoch)

                        # logging 
                        loss_cpu = info['losses']['action_loss'].item()
                        tepoch.set_postfix(loss=loss_cpu, refresh=False)
                        train_losses.append(loss_cpu)
                        step_log = {
                            'train_loss': loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch
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
                self.model.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(self.model)
                    # log all
                    # step_log.update(runner_log)
                    save_video(runner_log,local_epoch_idx)
                    
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
                                info = self.model.train_on_batch(batch, epoch=self.epoch, validate=True)
                                loss = info['losses']['action_loss']
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

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
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                self.model.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                # wandb_run.log(step_log, step=self.global_step)
                step_log['test_mean_score_last10'] = average_test_success.value
                step_log['train_mean_score_last10'] = average_train_success.value
                wandb_run.log(step_log, step=local_epoch_idx)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

'''

python diffusion_policy/run1/a12_lift_abs_DDPM100.py \
    --config-path=../config \
    --config-name=train_robomimic_lowdim_workspace.yaml \
    task=lift_lowdim_abs  


'''

'''

lift_lowdim_abs
lift
pusht_lowdim
square_lowdim_abs
tool_hang_lowdim_abs

python diffusion_policy/workspace/ryan_train_robomimic_lowdim_workspace.py \
    --config-path=../config \
    --config-name=train_robomimic_lowdim_workspace.yaml \
    task=lift_lowdim_abs  

conda install -c menpo osmesa -y
conda install -c conda-forge mujoco-py
export HYDRA_FULL_ERROR=1

conda install -c conda-forge patchelf

conda install conda-forge::libgcrypt

tar -xvzf storage_folder/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco

rm -rf ~/.mujoco/mujoco210/bin/_pyxbld*
python setup.py install

C_INCLUDE_PATH=:$HOME/.conda/envs/robodiff/include



python diffusion_policy/workspace/ryan_train_robomimic_lowdim_workspace.py  \
   --config-path=../config    \
--config-name=train_robomimic_lowdim_workspace.yaml \
task=pusht_lowdim

python diffusion_policy/run1/a12_lift_abs_DDPM100.py \
    --config-path=../config \
    --config-name=train_robomimic_lowdim_workspace.yaml \
    task=lift_lowdim_abs  



'''

import datetime

@hydra.main(
    version_base=None,
    # config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_path='../config',
    # config_name=pathlib.Path(__file__).stem)
    config_name='a12_robomimic_lowdim_can.yaml')
def main(cfg):
    # cfg.task='pusht_lowdim'
    # cfg.task.name='lift_lowdim_abs'
    # cfg.task.dataset.name='RobomimicLowdimDataset'
    # cfg.task.name='can_lowdim_abs'
    # cfg.task.name='square_lowdim_abs'
    # cfg.task.name='tool_hang_lowdim_abs'
    # cfg.task.name='transport_lowdim_abs'
    timestamp=datetime.datetime.now().strftime("%m%d%H%M-%S")
    workspace = TrainRobomimicLowdimWorkspace(cfg)
    


    
    script_name=os.path.basename(__file__).replace('.py','')
    # script_name = remove_prefix(script_name)

    wandb_name=f'{cfg.task.name}_{script_name}_{timestamp}'
    wandb_project_name='robomimic_lowdim_can'
    workspace.run(wandb_name,wandb_project_name)

if __name__ == "__main__":
    main()

