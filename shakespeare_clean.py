import pandas as pd
import numpy as np
import shutil
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

if __name__ == "__main__":
    folder_path = kagglehub.dataset_download("thedevastator/the-bards-best-a-character-modeling-dataset")

    # print("Files downloaded:", os.listdir(folder_path))

    train_df = pd.read_csv(os.path.join(folder_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(folder_path, "test.csv"))
    val_df = pd.read_csv(os.path.join(folder_path, "validation.csv"))

    train_df.to_csv("shakespeare_train.csv", index=False)
    test_df.to_csv("shakespeare_test.csv", index=False)
    val_df.to_csv("shakespeare_val.csv", index=False)

    # df = pd.concat([train_df, test_df, val_df], ignore_index=True)

    # df.to_csv("shakespeare.csv", index=False)

    print(f"Download Shakespears Dataset completed.")