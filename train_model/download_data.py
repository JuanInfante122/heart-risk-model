
import requests
import zipfile
import os

def download_and_extract_zip(url, save_path, extract_to):
    """
    Downloads a file from a URL, saves it locally, and extracts its contents.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The path to save the downloaded file.
        extract_to (str): The directory to extract the contents of the zip file.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=128):
                f.write(chunk)
        print(f"File downloaded successfully and saved to {save_path}")

        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"File extracted successfully to {extract_to}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

if __name__ == '__main__':
    # URL of the zip file from Kaggle
    url = 'https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/download?datasetVersionNumber=1'
    save_path = 'cardiovascular_disease.zip'
    extract_to = 'data/raw/'

    download_and_extract_zip(url, save_path, extract_to)
