import os
import subprocess
import wget
import ssl
import requests
import asyncio
import aiohttp
import zipfile
import os

ssl._create_default_https_context = ssl._create_unverified_context

def extract_zip(zip_file_path, extract_to):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def setup_kaggle_api():
    # Install Kaggle API
    subprocess.run(["pip", "install", "kaggle"])

    # Make directory for storing Kaggle API key
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    # Copy Kaggle API key file from repository
    kaggle_api_key_repo_path = "kaggle.json"  # Path to kaggle.json in your GitHub repo root
    kaggle_api_key_path = os.path.join(kaggle_dir, "kaggle.json")
    subprocess.run(["cp", kaggle_api_key_repo_path, kaggle_api_key_path])

    # Set permissions for Kaggle API key file
    os.chmod(kaggle_api_key_path, 0o600)

def download_dataset(username, dataset_name,file1,file2):
    # Download dataset using Kaggle API
    # subprocess.run(["kaggle", "datasets", "download", "-c", f"{dataset_name}","-f", f"{file1}","-f",f"{file2}"])
    subprocess.run(["kaggle", "competitions", "download",f"{dataset_name}","-f", f"{file2}"])
    # subprocess.run(["kaggle", "competitions", "download",f"{dataset_name}","-f", f"{file1}"])


def download_lesion_dataset(url):
    # wget.download(url, "A. Segmentation.zip")
    # response = requests.get(url)
    # with open("A. Segmentation.zip", "wb") as file:
    #     file.write(response.content)
    response = requests.get(url, stream=True)
    with open("A. Segmentation.zip", mode="wb") as file:
     for chunk in response.iter_content(chunk_size=10 * 1024):
        file.write(chunk)

async def download_file(url):
     async with aiohttp.ClientSession() as session:
         async with session.get(url) as response:
            if "content-disposition" in response.headers:
                header = response.headers["content-disposition"]
                filename = header.split("filename=")[1]
            else:
                filename = url.split("/")[-1]
                with open(filename, mode="wb") as file:
                 while True:
                    chunk = await response.content.read()
                    if not chunk:
                        break
                    file.write(chunk)
                 print(f"Downloaded file {filename}")
  

def main():
    # Set up Kaggle API
    setup_kaggle_api()

    # Define Kaggle dataset details
    kaggle_username = "bhaktibhanushaliii"
    dataset_name = "diabetic-retinopathy-detection"
    file1 = "train.zip.001"
    file2 = "trainLabels.csv.zip"

    # Download dataset
    download_dataset(kaggle_username, dataset_name,file1,file2)
    extract_zip(f"{file2}",'./')
    extract_zip(f"{file1}",'./')


    url = "https://ieee-dataport.s3.amazonaws.com/open/3754/A.%20Segmentation.zip?response-content-disposition=attachment%3B%20filename%3D%22A.%20Segmentation.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20240402%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240402T213242Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=322b502a2b186c51c0b88e2f9d8288e5589910fd55ca1aaad92eef4860bef094"

    # download_lesion_dataset(url)
    return 1

if __name__ == "__main__":
    main()
