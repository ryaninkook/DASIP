#!/bin/bash

python diffusion_policy/run2/a1_2_linear_robomimic_lowdim_abs_sample_only.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=SDE_CURVE \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=linear \
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083630_square_lowdim_abs_unet_SDE_CURVE_W1000_TS2/checkpoints \
    +policy.WSCALE=1000 \
    +policy.TS=2


python diffusion_policy/run2/a1_2_linear_robomimic_lowdim_abs_sample_only.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=SDE_CURVE \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=linear \
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083630_square_lowdim_abs_unet_SDE_CURVE_W1000_TS2/checkpoints \
    +policy.WSCALE=1000 \
    +policy.TS=4



python diffusion_policy/run2/a1_2_linear_robomimic_lowdim_abs_sample_only.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=SDE_CURVE \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=linear \
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083630_square_lowdim_abs_unet_SDE_CURVE_W1000_TS2/checkpoints \
    +policy.WSCALE=1000 \
    +policy.TS=10




python diffusion_policy/run2/a1_2_linear_robomimic_lowdim_abs_sample_only.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=SDE_CURVE \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=linear \
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083630_square_lowdim_abs_unet_SDE_CURVE_W1000_TS2/checkpoints \
    +policy.WSCALE=1000 \
    +policy.TS=50 



python diffusion_policy/run2/a1_2_linear_robomimic_lowdim_abs_sample_only.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_v3 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=SDE_CURVE \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=linear \
    +training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083630_square_lowdim_abs_unet_SDE_CURVE_W1000_TS2/checkpoints \
    +policy.WSCALE=1000 \
    +policy.TS=100