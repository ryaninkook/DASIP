#%%
import subprocess,os
def aws_upload_everything(from_folder,target_folder,DELETE=False,EXCLUDE=False,exclude_list=[]):
    exclude_list = [element for element in exclude_list if element != '']
    from_folder = from_folder.rstrip('/') + '/'
    target_folder = target_folder.rstrip('/') + '/'
    command =f"aws s3 sync {from_folder} s3://bluerand3/{target_folder} --exclude '.git/*'  --exclude '__pycache__/*' --exclude 'ignore/*'"
    for item in exclude_list:
        command=command+f" --exclude '{item}/*'"
    if EXCLUDE:
        command=command+" --exclude 'data/*'"
    if DELETE:
        command=command+' --delete'
    subprocess.run(command, shell=True)
    print('success')
from_folder=os.getcwd()
target_folder='robo/diffusion_policy'

list1=[
    'data',
    'media',
    'storage_folder',
    '.history',
    'aws',
    'mesa',
]
aws_upload_everything(from_folder,target_folder,EXCLUDE=True,exclude_list=list1)
# %%


#%%