import torch


def get_device():
    print(torch.cuda.is_available())  # Should return True if GPU is available
    print(torch.cuda.current_device())  # Prints the current CUDA device (if any)
    print(torch.cuda.get_device_name(0))  # Get the name of the GPU
    return "cuda" if torch.cuda.is_available() else "cpu"
