
python diffusion_policy/run2/a1_3_GVP_robomimic_lowdim_abs_sample_only.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_GVP_v4_new_train \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.ODE_SDE=ODE \
    +policy.DDPM=Euler \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=GVP \
    +training.save_path=data/output2/a15_7_unet_GVP_abs/20240927_103128_tool_hang_lowdim_abs_unet_euler_W100_TS2/checkpoints \
    +policy.diffusion_form=None \
    +policy.WSCALE=1000 \
    +policy.TS=1

python diffusion_policy/run2/a1_3_GVP_robomimic_lowdim_abs_sample_only.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_GVP_v4_new_train \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.ODE_SDE=ODE \
    +policy.DDPM=Euler \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=GVP \
    +training.save_path=data/output2/a15_7_unet_GVP_abs/20240927_103128_tool_hang_lowdim_abs_unet_euler_W100_TS2/checkpoints \
    +policy.diffusion_form=None \
    +policy.WSCALE=1000 \
    +policy.TS=2



python diffusion_policy/run2/a1_3_GVP_robomimic_lowdim_abs_sample_only.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_GVP_v4_new_train \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.ODE_SDE=ODE \
    +policy.DDPM=Euler \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=GVP \
    +training.save_path=data/output2/a15_7_unet_GVP_abs/20240927_103128_tool_hang_lowdim_abs_unet_euler_W100_TS2/checkpoints \
    +policy.diffusion_form=None \
    +policy.WSCALE=1000 \
    +policy.TS=4


python diffusion_policy/run2/a1_3_GVP_robomimic_lowdim_abs_sample_only.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_GVP_v4_new_train \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.ODE_SDE=ODE \
    +policy.DDPM=Euler \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=GVP \
    +training.save_path=data/output2/a15_7_unet_GVP_abs/20240927_103128_tool_hang_lowdim_abs_unet_euler_W100_TS2/checkpoints \
    +policy.diffusion_form=None \
    +policy.WSCALE=1000 \
    +policy.TS=10


python diffusion_policy/run2/a1_3_GVP_robomimic_lowdim_abs_sample_only.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_GVP_v4_new_train \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.ODE_SDE=ODE \
    +policy.DDPM=Euler \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=GVP \
    +training.save_path=data/output2/a15_7_unet_GVP_abs/20240927_103128_tool_hang_lowdim_abs_unet_euler_W100_TS2/checkpoints \
    +policy.diffusion_form=None \
    +policy.WSCALE=1000 \
    +policy.TS=50


python diffusion_policy/run2/a1_3_GVP_robomimic_lowdim_abs_sample_only.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_GVP_v4_new_train \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.ODE_SDE=ODE \
    +policy.DDPM=Euler \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=GVP \
    +training.save_path=data/output2/a15_7_unet_GVP_abs/20240927_103128_tool_hang_lowdim_abs_unet_euler_W100_TS2/checkpoints \
    +policy.diffusion_form=None \
    +policy.WSCALE=1000 \
    +policy.TS=100







