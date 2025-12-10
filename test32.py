import os
from pathlib import Path


folder_name = r"D:\Downloads\acdc\labels\fog"
folder = Path(folder_name)

for file_name in os.listdir(folder_name):
    file = folder/file_name
    file_renamed = folder / (file_name.replace('_gt_labelColor', ''))
    if file.is_file():
        os.rename(file.resolve(), file_renamed.resolve())