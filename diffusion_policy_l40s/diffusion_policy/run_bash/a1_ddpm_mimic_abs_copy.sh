#!/bin/bash
# chmod +x diffusion_policy/run_bash/a1_ddpm_mimic_abs_copy.sh
# ./diffusion_policy/run_bash/a1_ddpm_mimic_abs_copy.sh


# Lift - TS=3
gnome-terminal -- bash -c "python diffusion_policy/run1/a17_unet_DDPM_mimic_abs.py \
    --config-path=../config \
    --config-name=ryan_unet_lowdim_lift_lowdim_abs.yaml \
    task=lift_lowdim_abs  \
    logging.project=unet_ddpm_lift_abs \
    logging.name=lift_ddpm_ts3 \
    +training.num_workers=0 \
    +training.TS=3; exec bash"

# Tool Hang - TS=3
gnome-terminal -- bash -c "python diffusion_policy/run1/a17_unet_DDPM_mimic_abs.py \
    --config-path=../config \
    --config-name=ryan_unet_lowdim_tool_hang_lowdim_abs.yaml \
    task=tool_hang_lowdim_abs  \
    logging.project=unet_ddpm_tool_hang_abs \
    logging.name=tool_hang_ddpm_ts3 \
    +training.num_workers=0 \
    +training.TS=3; exec bash"

# Transport - TS=3
gnome-terminal -- bash -c "python diffusion_policy/run1/a17_unet_DDPM_mimic_abs.py \
    --config-path=../config \
    --config-name=ryan_unet_lowdim_transport_lowdim_abs.yaml \
    task=transport_lowdim_abs  \
    logging.project=unet_ddpm_transport_abs \
    logging.name=transport_ddpm_ts3 \
    +training.num_workers=0 \
    +training.TS=3; exec bash"

# Square - TS=3
gnome-terminal -- bash -c "python diffusion_policy/run1/a17_unet_DDPM_mimic_abs.py \
    --config-path=../config \
    --config-name=ryan_unet_lowdim_square_lowdim_abs.yaml \
    task=square_lowdim_abs  \
    logging.project=unet_ddpm_square_abs \
    logging.name=square_ddpm_ts3 \
    +training.num_workers=0 \
    +training.TS=3; exec bash"

# Can - TS=3
gnome-terminal -- bash -c "python diffusion_policy/run1/a17_unet_DDPM_mimic_abs.py \
    --config-path=../config \
    --config-name=ryan_unet_lowdim_can_lowdim_abs.yaml \
    task=can_lowdim_abs  \
    logging.project=unet_ddpm_can_abs \
    logging.name=can_ddpm_ts3 \
    +training.num_workers=0 \
    +training.TS=3; exec bash"
