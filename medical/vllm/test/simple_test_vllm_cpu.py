import sys
import os

"""
简单的vLLM CPU模式测试脚本
直接测试vLLM是否能在CPU模式下工作
"""


def main():
    print("="*60)
    print("vLLM CPU模式简单测试")
    print("="*60)
    
    # 尝试导入vllm
    try:
        print("尝试导入vllm...")
        from vllm import LLM, SamplingParams
        print("成功导入vllm！")
        
        # 配置采样参数
        sampling_params = SamplingParams(temperature=0.1, max_tokens=100)
        print(f"采样参数: {sampling_params}")
        
        # 尝试使用小模型初始化LLM，强制使用CPU
        print("\n尝试使用gpt2模型初始化LLM（CPU模式）...")
        try:
            llm = LLM(
                model="gpt2",
                device="cpu",
                max_model_len=1024,
                trust_remote_code=True
            )
            print("LLM模型初始化成功！")
            
            # 测试简单推理
            print("\n测试简单推理...")
            prompt = "什么是依赖管理？请简要解释。"
            outputs = llm.generate([prompt], sampling_params)
            
            # 输出结果
            print(f"\n推理结果:")
            print(outputs[0].outputs[0].text)
            
            print("\n" + "="*60)
            print("✅ 测试成功！vllm在CPU模式下工作正常。")
            print("="*60)
            
        except Exception as model_error:
            print(f"LLM初始化失败: {str(model_error)}")
            import traceback
            traceback.print_exc()
            print("\n" + "="*60)
            print("❌ LLM初始化失败。")
            print("\n故障排查建议:")
            print("1. 确保您的系统有足够的内存")
            print("2. 检查网络连接是否正常（需要下载模型）")
            print("3. 尝试使用本地已下载的模型路径")
            print("="*60)
            
    except ImportError as import_error:
        print(f"vllm导入失败: {str(import_error)}")
        print("\n" + "="*60)
        print("❌ vllm导入失败。")
        print("\n建议执行以下命令重新安装vllm:")
        print(f"{sys.executable} -m pip install vllm --upgrade --no-cache-dir")
        print("="*60)
    
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        print("❌ 测试失败。")
        print("="*60)


if __name__ == "__main__":
    main()