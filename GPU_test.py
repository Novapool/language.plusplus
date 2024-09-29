import torch
import os

def test_gpu():
    print("CUDA Environment Variable:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
    print("\nPyTorch CUDA Information:")
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"\nNumber of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nCUDA Device {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
        print(f"\nCurrent CUDA device index: {torch.cuda.current_device()}")
        print(f"Current CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        # Test with a small tensor operation
        print("\nTesting with a small tensor operation:")
        a = torch.cuda.FloatTensor(2).zero_()
        print(f"Tensor a: {a}")
        print(f"Tensor a is on CUDA: {a.is_cuda}")
        print(f"Tensor a's device: {a.device}")
    else:
        print("\nNo CUDA device available. PyTorch will use CPU.")

if __name__ == "__main__":
    test_gpu()