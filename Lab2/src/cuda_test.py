import torch

def main():
    print('CUDA Test')
    print(f'CUDA available: {torch.cuda.is_available()}')
    
    if torch.cuda.is_available():
        print(f'CUDA device count: {torch.cuda.device_count()}')
        print(f'Current CUDA device: {torch.cuda.current_device()}')
        print(f'CUDA device name: {torch.cuda.get_device_name(0)}')
        print(f'CUDA version: {torch.version.cuda}')
    else:
        print('CUDA is not available. Using CPU.')

if __name__ == '__main__':
    main()
