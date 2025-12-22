#!/bin/bash
# chmod +x diffusion_policy/run_bash/a1_ddpm_mimic_abs.sh
# ./diffusion_policy/run_bash/a1_ddpm_mimic_abs.sh


# Lift - TS=3
code --new-window \
    --add . \
    --command workbench.action.terminal.newInActiveWorkspace; \
    sleep 1;
code --add . \
    --command "workbench.action.terminal.sendSequence" \
    --args "{\"text\": \"python diffusion_policy/run1/a17_unet_DDPM_mimic_abs.py \\
    --config-path=../config \\
    --config-name=ryan_unet_lowdim_lift_lowdim_abs.yaml \\
    task=lift_lowdim_abs  \\
    logging.project=unet_ddpm_lift_abs \\
    logging.name=lift_ddpm_ts3 \\
    +training.num_workers=0 \\
    +training.TS=3\\n\"}"; \
    sleep 1;

# Lift - TS=1
code --add . \
    --command workbench.action.terminal.newInActiveWorkspace; \
    sleep 1;
code --add . \
    --command "workbench.action.terminal.sendSequence" \
    --args "{\"text\": \"python diffusion_policy/run1/a17_unet_DDPM_mimic_abs.py \\
    --config-path=../config \\
    --config-name=ryan_unet_lowdim_lift_lowdim_abs.yaml \\
    task=lift_lowdim_abs  \\
    logging.project=unet_ddpm_lift_abs \\
    logging.name=lift_ddpm_ts1 \\
    +training.num_workers=0 \\
    +training.TS=1\\n\"}"; \
    sleep 1;

# Tool Hang - TS=3
code --add . \
    --command workbench.action.terminal.newInActiveWorkspace; \
    sleep 1;
code --add . \
    --command "workbench.action.terminal.sendSequence" \
    --args "{\"text\": \"python diffusion_policy/run1/a17_unet_DDPM_mimic_abs.py \\
    --config-path=../config \\
    --config-name=ryan_unet_lowdim_tool_hang_lowdim_abs.yaml \\
    task=tool_hang_lowdim_abs  \\
    logging.project=unet_ddpm_tool_hang_abs \\
    logging.name=tool_hang_ddpm_ts3 \\
    +training.num_workers=0 \\
    +training.TS=3\\n\"}"; \
    sleep 1;

# Tool Hang - TS=1
code --add . \
    --command workbench.action.terminal.newInActiveWorkspace; \
    sleep 1;
code --add . \
    --command "workbench.action.terminal.sendSequence" \
    --args "{\"text\": \"python diffusion_policy/run1/a17_unet_DDPM_mimic_abs.py \\
    --config-path=../config \\
    --config-name=ryan_unet_lowdim_tool_hang_lowdim_abs.yaml \\
    task=tool_hang_lowdim_abs  \\
    logging.project=unet_ddpm_tool_hang_abs \\
    logging.name=tool_hang_ddpm_ts1 \\
    +training.num_workers=0 \\
    +training.TS=1\\n\"}"; \
    sleep 1;

# Transport - TS=3
code --add . \
    --command workbench.action.terminal.newInActiveWorkspace; \
    sleep 1;
code --add . \
    --command "workbench.action.terminal.sendSequence" \
    --args "{\"text\": \"python diffusion_policy/run1/a17_unet_DDPM_mimic_abs.py \\
    --config-path=../config \\
    --config-name=ryan_unet_lowdim_transport_lowdim_abs.yaml \\
    task=transport_lowdim_abs  \\
    logging.project=unet_ddpm_transport_abs \\
    logging.name=transport_ddpm_ts3 \\
    +training.num_workers=0 \\
    +training.TS=3\\n\"}"; \
    sleep 1;

# Transport - TS=1
code --add . \
    --command workbench.action.terminal.newInActiveWorkspace; \
    sleep 1;
code --add . \
    --command "workbench.action.terminal.sendSequence" \
    --args "{\"text\": \"python diffusion_policy/run1/a17_unet_DDPM_mimic_abs.py \\
    --config-path=../config \\
    --config-name=ryan_unet_lowdim_transport_lowdim_abs.yaml \\
    task=transport_lowdim_abs  \\
    logging.project=unet_ddpm_transport_abs \\
    logging.name=transport_ddpm_ts1 \\
    +training.num_workers=0 \\
    +training.TS=1\\n\"}"; \
    sleep 1;

# Square - TS=3
code --add . \
    --command workbench.action.terminal.newInActiveWorkspace; \
    sleep 1;
code --add . \
    --command "workbench.action.terminal.sendSequence" \
    --args "{\"text\": \"python diffusion_policy/run1/a17_unet_DDPM_mimic_abs.py \\
    --config-path=../config \\
    --config-name=ryan_unet_lowdim_square_lowdim_abs.yaml \\
    task=square_lowdim_abs  \\
    logging.project=unet_ddpm_square_abs \\
    logging.name=square_ddpm_ts3 \\
    +training.num_workers=0 \\
    +training.TS=3\\n\"}"; \
    sleep 1;

# Square - TS=1
code --add . \
    --command workbench.action.terminal.newInActiveWorkspace; \
    sleep 1;
code --add . \
    --command "workbench.action.terminal.sendSequence" \
    --args "{\"text\": \"python diffusion_policy/run1/a17_unet_DDPM_mimic_abs.py \\
    --config-path=../config \\
    --config-name=ryan_unet_lowdim_square_lowdim_abs.yaml \\
    task=square_lowdim_abs  \\
    logging.project=unet_ddpm_square_abs \\
    logging.name=square_ddpm_ts1 \\
    +training.num_workers=0 \\
    +training.TS=1\\n\"}"; \
    sleep 1;

# Can - TS=3
code --add . \
    --command workbench.action.terminal.newInActiveWorkspace; \
    sleep 1;
code --add . \
    --command "workbench.action.terminal.sendSequence" \
    --args "{\"text\": \"python diffusion_policy/run1/a17_unet_DDPM_mimic_abs.py \\
    --config-path=../config \\
    --config-name=ryan_unet_lowdim_can_lowdim_abs.yaml \\
    task=can_lowdim_abs  \\
    logging.project=unet_ddpm_can_abs \\
    logging.name=can_ddpm_ts3 \\
    +training.num_workers=0 \\
    +training.TS=3\\n\"}"; \
    sleep 1;

# Can - TS=1
code --add . \
    --command workbench.action.terminal.newInActiveWorkspace; \
    sleep 1;
code --add . \
    --command "workbench.action.terminal.sendSequence" \
    --args "{\"text\": \"python diffusion_policy/run1/a17_unet_DDPM_mimic_abs.py \\
    --config-path=../config \\
    --config-name=ryan_unet_lowdim_can_lowdim_abs.yaml \\
    task=can_lowdim_abs  \\
    logging.project=unet_ddpm_can_abs \\
    logging.name=can_ddpm_ts1 \\
    +training.num_workers=0 \\
    +training.TS=1\\n\"}"; \
    sleep 1;
