#!/bin/bash

# Content from vp_sde_decreasing/blockpush_0.sh
python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061613_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=100



# Content from vp_sde_decreasing/tool_hang_1.sh
python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061559_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=100



# Content from vp_sde_decreasing/transport_2.sh
python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061602_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=100



# Content from vp_sde_decreasing/square_3.sh
python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061606_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=100



# Content from vp_sde_decreasing/kitchen_4.sh
export MUJOCO_GL=egl

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061626_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=100



# Content from vp_sde_decreasing/blockpush_7.sh
python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061613_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=1

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061613_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=2

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061613_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=3

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061613_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=4

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061613_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=5

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061613_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061613_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=25

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061613_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=50



# Content from vp_sde_decreasing/tool_hang_9.sh
python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061559_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=1

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061559_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=2

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061559_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=3

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061559_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=4

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061559_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=5

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061559_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061559_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=25

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061559_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=50



# Content from vp_sde_decreasing/transport_12.sh
python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061602_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=1

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061602_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=2

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061602_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=3

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061602_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=4

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061602_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=5

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061602_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061602_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=25

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=unet_transport_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061602_transport_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=50



# Content from vp_sde_decreasing/square_15.sh
python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061606_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=1

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061606_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=2

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061606_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=3

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061606_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=4

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061606_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=5

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061606_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061606_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=25

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061606_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=50



# Content from vp_sde_decreasing/kitchen_18.sh
export MUJOCO_GL=egl

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061626_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=1

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061626_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=2

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061626_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=3

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061626_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=4

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061626_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=5

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061626_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061626_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=25

python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim_abs  \
    logging.name=kitchen_lowdim_abs \
    logging.project=unet_kitchen_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061626_kitchen_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=50



