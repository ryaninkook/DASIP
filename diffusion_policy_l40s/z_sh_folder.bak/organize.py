#%%
import os

os.getcwd()

# do cd .. on current getcwd
os.getcwd().split('/')[:-1]
from_folder='/'.join(os.getcwd().split('/')[:-1])
from_folder
#z_sh_folder
to_folder=os.path.join(from_folder,'z_sh_folder')
print("to_folder: ",to_folder)
# Define the folder paths
# from_folder = '/mnt/H1/code/robo/LIBERO/a1_3_debug_dataset.py'
# to_folder = '/mnt/H1/code/robo/LIBERO/z_sh_folder'

# Get list of files in the 'from_folder'
from_files = [f for f in os.listdir(from_folder) if f.endswith('.sh')]
from_files
# print("from_files: ",from_files)

selected=[
    
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9
          ]

# convert selected to 1.sh and 5.sh
selected=[f'{i}.sh' for i in selected]
selected
# from_files
# from_files=[from_files[i-1] for i in selected]
# from_files
# print("from_files: ",from_files)
print("selected: ",selected)
# Process each file
for filename in selected:
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
# %%
