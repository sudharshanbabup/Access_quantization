import os
import ssl
import tarfile
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

def download_cifar10(dest_dir):
    print("Downloading CIFAR-10 dataset from fast.ai S3 mirror...")
    tgz_path = os.path.join(dest_dir, "cifar10.tgz")
    url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    
    if not os.path.exists(os.path.join(dest_dir, "cifar10")):
        if not os.path.exists(tgz_path):
            print(f"Downloading {url} to {tgz_path}...")
            urllib.request.urlretrieve(url, tgz_path)
            print("Download complete.")
        else:
            print("tgz file already exists.")
            
        print("Extracting CIFAR-10...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
        print("CIFAR-10 extracted successfully.")
        
        if os.path.exists(tgz_path):
            os.remove(tgz_path)
            print("Cleaned up tgz archive.")
    else:
        print("CIFAR-10 already extracted.")

def download_imagenette(dest_dir):
    print("Downloading ImageNette (160px version) from fast.ai S3...")
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    tgz_path = os.path.join(dest_dir, "imagenette2-160.tgz")
    
    if not os.path.exists(os.path.join(dest_dir, "imagenette2-160")):
        if not os.path.exists(tgz_path):
            print(f"Downloading {url} to {tgz_path}...")
            urllib.request.urlretrieve(url, tgz_path)
            print("Download complete.")
        else:
            print("tgz file already exists.")
            
        print("Extracting ImageNette...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
        print("ImageNette extracted successfully.")
        
        if os.path.exists(tgz_path):
            os.remove(tgz_path)
            print("Cleaned up tgz archive file.")
    else:
        print("ImageNette already extracted.")

if __name__ == "__main__":
    base_datasets_dir = "/Users/sudharshanbabupandava/JioCloud/CMR University/Research/Ravi Saidala/Ravi_Saidala_v3/datasets_v3"
    os.makedirs(base_datasets_dir, exist_ok=True)
    
    download_cifar10(base_datasets_dir)
    download_imagenette(base_datasets_dir)
    print("All dataset downloads completed!")
