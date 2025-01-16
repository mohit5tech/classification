import os

def download_dataset(kaggle_dataset, download_path):
    """Downloads and extracts a Kaggle dataset."""
    # Ensure the Kaggle API is installed
    os.system("pip install kaggle")

    # Create the directory if it doesn't exist
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Download and unzip the dataset
    os.system(f"kaggle datasets download -d {kaggle_dataset} -p {download_path} --unzip")
    print(f"Dataset downloaded and extracted to: {download_path}")

# Example usage:
if __name__ == '__main__':
    download_dataset(kaggle_dataset="paultimothymooney/chest-xray-pneumonia", download_path="datasets/chest_xray_pneumonia")
