import os

source_dir = "../data/CASF-2016/coreset"
target_dir = "../data/complexes_16"

# find all folder names in the source directory
folder_names = [name for name in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, name))]

# create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)
# create new folders in the target directory
for folder in folder_names:
    new_folder_path = os.path.join(target_dir, folder)
    os.makedirs(new_folder_path, exist_ok=True)
    print(f"Created: {new_folder_path}")