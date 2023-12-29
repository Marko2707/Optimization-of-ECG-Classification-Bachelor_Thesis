import os

def is_folder_empty(folder):
    #Takes in content of given folder
    content = os.listdir(folder)

    #returns boolean if folder is empty
    return len(content) == 0