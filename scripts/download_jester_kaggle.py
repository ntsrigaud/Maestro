import os
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

def download_jester():
    api = KaggleApi()
    api.authenticate()
    
    dataset = "sanjanatg26/20bn-jester-v1-complete"
    target_dir = "./jester_dataset"
    
    print(f"Starting download for {dataset}...")
    
    # Using the low-level download to hook into the progress bar
    api.dataset_download_files(
        dataset, 
        path=target_dir, 
        unzip=True, 
        quiet=False # This enables the built-in Kaggle progress bar
    )

if __name__ == "__main__":
    download_jester()
