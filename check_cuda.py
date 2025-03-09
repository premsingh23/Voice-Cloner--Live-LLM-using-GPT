import torch
import sys
from utils import check_cuda_availability

def check_cuda():
    """
    Check if CUDA is available and print device information.
    Returns True if CUDA is available, False otherwise.
    """
    device, device_info = check_cuda_availability()
    
    if device == "cuda":
        print(f"CUDA is available! Using device: {device_info['name']}")
        print(f"CUDA version: {device_info['cuda_version']}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Number of available CUDA devices: {device_info['count']}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        if device_info['cudnn_version']:
            print(f"cuDNN version: {device_info['cudnn_version']}")
        return True
    else:
        print("CUDA is not available. Using CPU instead.")
        print(f"PyTorch version: {torch.__version__}")
        return False

if __name__ == "__main__":
    # When run directly, print CUDA info and exit with status code
    # 0 if CUDA is available, 1 if not
    if check_cuda():
        sys.exit(0)
    else:
        sys.exit(1) 