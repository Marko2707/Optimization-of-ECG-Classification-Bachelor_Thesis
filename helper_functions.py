import os

def is_folder_empty(folder):
    #Takes in content of given folder
    content = os.listdir(folder)

    #returns boolean if folder is empty
    return len(content) == 0

import os

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")
