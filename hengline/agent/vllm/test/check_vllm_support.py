import torch

def check_vllm_support():
    has_cuda = torch.cuda.is_available()
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if has_cuda else 0
    cpu_cores = torch.get_num_threads()
    print(f"CUDA 可用: {has_cuda}")
    if has_cuda:
        print(f"GPU 显存: {gpu_mem:.2f} GB")
    print(f"CPU 核心数: {cpu_cores}")

    if not has_cuda:
        print("警告：vLLM 主要依赖 CUDA 和 NVIDIA GPU，无 CUDA 时可能无法运行。")
    elif gpu_mem < 6:
        print("警告：显存小于 6GB，可能无法运行大多数 vLLM 支持的大模型。")
    else:
        print("硬件基本满足 vLLM 运行要求，但低配笔记本仍可能遇到性能瓶颈。")

check_vllm_support()
