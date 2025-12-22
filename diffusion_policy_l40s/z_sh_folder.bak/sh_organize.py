#%%
import os

# Define the folder paths
from_folder = '/data/ryan/code/robo/diffusion_policy'
to_folder = '/data/ryan/code/robo/diffusion_policy/z_sh_folder'

# Get list of files in the 'from_folder'
from_files = [f for f in os.listdir(from_folder) if f.endswith('.sh')]
from_files
#%%

# Process each file
for filename in from_files:
    from_file_path = os.path.join(from_folder, filename)
    to_file_path = os.path.join(to_folder, filename)
    
    # Check if the file exists in the 'to_folder', if not, create it
    if not os.path.exists(to_file_path):
        with open(to_file_path, 'w') as f:
            pass  # Create an empty file if it doesn't exist
    
    # Read content from 'from_folder' file
    with open(from_file_path, 'r') as f:
        from_content = f.read()
    
    # Append the content to the corresponding file in 'to_folder' with ### and 3 newlines
    with open(to_file_path, 'a') as f:
        f.write("###\n")  # Add ### at the top of this paste
        f.write("\n\n\n")  # Add 3 newlines
        f.write(from_content)
        f.write("\n\n\n")  # Add 3 newlines
    
    # Clear the content of the file in 'from_folder'
    with open(from_file_path, 'w') as f:
        f.write('')

print("Files processed successfully.")

#%%

