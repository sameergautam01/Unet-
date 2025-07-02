import os

# Ask user for the folder path
folder_path = input("Enter the path to your folder: ").strip('"').strip("'")

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if '_class0' in filename:
        name, ext = os.path.splitext(filename)
        if name.endswith('_class0'):
            new_name = name[:-7] + ext  # Remove '_class0' from base name
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} -> {new_name}')