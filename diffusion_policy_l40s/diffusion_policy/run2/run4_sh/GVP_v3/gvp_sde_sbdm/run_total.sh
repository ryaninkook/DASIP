#!/bin/bash

# Content from gvp_sde_sbdm/blockpush_0.sh
python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241001_070116_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=100



# Content from gvp_sde_sbdm/tool_hang_1.sh
python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_2_GVP_new_v3_without_eval/20241013_061317_tool_hang_lowdim_abs_unet_euler_W1000_TS1/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=100



# Content from gvp_sde_sbdm/transport_2.sh
python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_100454_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=100



# Content from gvp_sde_sbdm/square_3.sh
python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_141443_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=100



# Content from gvp_sde_sbdm/kitchen_4.sh
export MUJOCO_GL=egl

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241002_152653_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=100



# Content from gvp_sde_sbdm/blockpush_7.sh
python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241001_070116_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=1

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241001_070116_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=2

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241001_070116_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=3

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241001_070116_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=4

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241001_070116_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=5

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241001_070116_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241001_070116_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=25

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241001_070116_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=50



# Content from gvp_sde_sbdm/tool_hang_9.sh
python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_2_GVP_new_v3_without_eval/20241013_061317_tool_hang_lowdim_abs_unet_euler_W1000_TS1/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=1

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_2_GVP_new_v3_without_eval/20241013_061317_tool_hang_lowdim_abs_unet_euler_W1000_TS1/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=2

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_2_GVP_new_v3_without_eval/20241013_061317_tool_hang_lowdim_abs_unet_euler_W1000_TS1/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=3

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_2_GVP_new_v3_without_eval/20241013_061317_tool_hang_lowdim_abs_unet_euler_W1000_TS1/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=4

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_2_GVP_new_v3_without_eval/20241013_061317_tool_hang_lowdim_abs_unet_euler_W1000_TS1/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=5

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_2_GVP_new_v3_without_eval/20241013_061317_tool_hang_lowdim_abs_unet_euler_W1000_TS1/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_2_GVP_new_v3_without_eval/20241013_061317_tool_hang_lowdim_abs_unet_euler_W1000_TS1/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=25

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_2_GVP_new_v3_without_eval/20241013_061317_tool_hang_lowdim_abs_unet_euler_W1000_TS1/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=50



# Content from gvp_sde_sbdm/transport_12.sh
python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_100454_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=1

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_100454_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=2

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_100454_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=3

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_100454_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=4

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_100454_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=5

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_100454_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_100454_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=25

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_100454_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=50



# Content from gvp_sde_sbdm/square_15.sh
python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_141443_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=1

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_141443_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=2

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_141443_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=3

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_141443_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=4

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_141443_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=5

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_141443_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_141443_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=25

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20240930_141443_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=50



# Content from gvp_sde_sbdm/kitchen_18.sh
export MUJOCO_GL=egl

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241002_152653_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=1

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241002_152653_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=2

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241002_152653_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=3

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241002_152653_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=4

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241002_152653_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=5

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241002_152653_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241002_152653_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=25

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_GVP_SDE_Euler_SBDM_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241002_152653_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=SBDM \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=50



