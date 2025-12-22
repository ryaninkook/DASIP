#!/bin/bash


python diffusion_policy/run2/a1_DDPM_robomimic_lowdim_abs_sample_only.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace_ryan_ddim.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_DDPM_v4 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=DDIM \
    +policy.T=100 \
    +training.save_50=True \
    +policy.path_type=DDIM \
    +training.save_path=data/output2/a15_6_unet_linear_abs_DDPM/20240923_084838_blockpush_lowdim_seed_abs_unet_DDPM_W100_TS2/checkpoints \
    +policy.WSCALE=1000 \
    +policy.TS=1

python diffusion_policy/run2/a1_DDPM_robomimic_lowdim_abs_sample_only.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace_ryan_ddim.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_DDPM_v4 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=DDIM \
    +policy.T=100 \
    +training.save_50=True \
    +policy.path_type=DDIM \
    +training.save_path=data/output2/a15_6_unet_linear_abs_DDPM/20240923_084838_blockpush_lowdim_seed_abs_unet_DDPM_W100_TS2/checkpoints \
    +policy.WSCALE=1000 \
    +policy.TS=2


python diffusion_policy/run2/a1_DDPM_robomimic_lowdim_abs_sample_only.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace_ryan_ddim.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_DDPM_v4 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=DDIM \
    +policy.T=100 \
    +training.save_50=True \
    +policy.path_type=DDIM \
    +training.save_path=data/output2/a15_6_unet_linear_abs_DDPM/20240923_084838_blockpush_lowdim_seed_abs_unet_DDPM_W100_TS2/checkpoints \
    +policy.WSCALE=1000 \
    +policy.TS=4



python diffusion_policy/run2/a1_DDPM_robomimic_lowdim_abs_sample_only.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace_ryan_ddim.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_DDPM_v4 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=DDIM \
    +policy.T=100 \
    +training.save_50=True \
    +policy.path_type=DDIM \
    +training.save_path=data/output2/a15_6_unet_linear_abs_DDPM/20240923_084838_blockpush_lowdim_seed_abs_unet_DDPM_W100_TS2/checkpoints \
    +policy.WSCALE=1000 \
    +policy.TS=10



