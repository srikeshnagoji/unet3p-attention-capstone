import pandas as pd
import os

def get_dataset_dataframe(base_path:str):
    data = []

    for dir_ in os.listdir(base_path):
        dir_path = os.path.join(base_path, dir_)
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                img_path = os.path.join(dir_path, filename)
                data.append([dir_, img_path])
        else:
            print(f"[INFO] This is not a dir --> {dir_path}")
            
    return pd.DataFrame(data, columns=["dir_name", "image_path"])