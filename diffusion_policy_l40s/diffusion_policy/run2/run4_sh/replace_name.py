#%%
import os
import re

# Define the directory where your .sh files are located

from_name = 'blockpush'
to_name = 'square'
directory = f'DDPM/{to_name}'
# from_checkpoint='+training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083622_transport_lowdim_abs_unet_SDE_CURVE_W1000_TS2/checkpoints'
# to_checkpoint='+training.save_path=data/output2/a15_3_unet_linear_interpol_mimic_abs/20240923_083626_transport_lowdim_abs_unet_SDE_CURVE_W1000_TS2/checkpoints'
from_checkpoint = 'data/output2/a15_6_unet_linear_abs_DDPM/20240923_084838_blockpush_lowdim_seed_abs_unet_DDPM_W100_TS2/checkpoints'
to_checkpoint = 'data/output2/a15_6_unet_linear_abs_DDPM/20240923_084831_square_lowdim_abs_unet_DDPM_W100_TS2/checkpoints'

from_checkpoint=from_checkpoint.replace(from_name,to_name)
# Loop through all files in the directory
for filename in os.listdir(directory):
    # Full path of the current file
    file_path = os.path.join(directory, filename)
    
    # Only process .sh files
    if filename.endswith('.sh'):
        # Replace content inside the file
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Replace 'tool_hang' with 'transport' in file content
        new_content = re.sub(from_name, to_name, content)
        
        # Replace the checkpoint path in file content
        new_content = re.sub(from_checkpoint, to_checkpoint, new_content)
        
        # If the content was modified, overwrite the file with the new content
        if new_content != content:
            with open(file_path, 'w') as file:
                file.write(new_content)
            print(f"Modified content in: {file_path}")
        
        # Rename the file if 'tool_hang' is in the filename
        if from_name in filename:
            new_filename = filename.replace(from_name, to_name)
            new_file_path = os.path.join(directory, new_filename)
            os.rename(file_path, new_file_path)
            print(f"Renamed: {file_path} -> {new_file_path}")

print("Replacement complete!")

# %%


#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%####################################################
#
import os

# Define the directories and files to process
folders = ["blockpush", "kitchen", "square",'tool_hang','transport']
base_path = "/data/ryan/code/robo/diffusion_policy/diffusion_policy/run2/run4_sh/linear_v3"
old_seed = "200000"
new_seed = "400000"

# Loop over each folder to create and modify files
for folder in folders:
    src_file = os.path.join(base_path, folder, f"ode123_seed{old_seed}.sh")
    dest_file = os.path.join(base_path, folder, f"ode123_seed{new_seed}.sh")
    
    # Read the content of the source file
    with open(src_file, 'r') as file:
        content = file.read()
    
    # Replace the old seed with the new seed
    new_content = content.replace(old_seed, new_seed)
    
    # Write the new content to the destination file
    with open(dest_file, 'w') as file:
        file.write(new_content)
    
    print(f"Created and updated {dest_file}")

# The code assumes the folders and files exist in the provided path. 
# You can change the base_path to test in a different directory.

# %%
#%%
#%%
import os

# Define the directories and files to process for run.sh creation
folders = ["blockpush", "kitchen", "square", "tool_hang", "transport"]
base_path = "/data/ryan/code/robo/diffusion_policy/diffusion_policy/run2/run4_sh/linear_v3"
seeds = ["200000", "300000", "400000"]

# Loop over each folder to create the run.sh file
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    run_file = os.path.join(folder_path, "run.sh")
    
    # Ensure the directory exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Create the content for run.sh
    content = "#!/bin/bash\n\n"
    content += "# Run ode1.sh\n\n"
    
    for seed in seeds:
        relative_path = f"diffusion_policy/run2/run4_sh/linear_v3/{folder}/ode123_seed{seed}.sh"
        content += f"# Run ode{seeds.index(seed) + 1}.sh\n"
        content += f"bash {relative_path}\n\n"
    
    # Write the content to run.sh
    with open(run_file, 'w') as file:
        file.write(content)
    
    print(f"Created {run_file}")



# %%
