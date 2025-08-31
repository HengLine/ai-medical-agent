import sys
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('raw_vllm_test')


def main():
    """
    直接测试原生vllm库功能，绕过langchain_community
    """
    print("="*60)
    print("原生vllm库测试工具")
    print("="*60)
    
    # 打印Python环境信息
    print(f"Python解释器: {sys.executable}")
    print(f"Python版本: {sys.version}")
    
    # 检查vllm导入
    try:
        logger.info("尝试导入vllm包...")
        import vllm
        from vllm import LLM, SamplingParams
        
        logger.info(f"成功导入vllm包，版本: {vllm.__version__}")
        print(f"\nvllm包信息:")
        print(f"- 版本: {vllm.__version__}")
        print(f"- 安装位置: {vllm.__file__}")
        
        # 测试VLLM的LLM类
        logger.info("测试VLLM的LLM类...")
        print("\n测试LLM类的可用性...")
        print(f"LLM类: {LLM}")
        print(f"SamplingParams类: {SamplingParams}")
        
        # 尝试使用小模型进行测试
        try:
            logger.info("尝试使用小型gpt2模型进行测试...")
            print("\n尝试加载小型gpt2模型...")
            
            # 创建采样参数
            sampling_params = SamplingParams(temperature=0.1, max_tokens=100)
            
            # 初始化LLM，使用较小的模型
            llm = LLM(model="gpt2", 
                     device="cpu",  # 强制使用CPU以避免GPU问题
                     max_model_len=1024,
                     trust_remote_code=True)
            
            logger.info("模型加载成功，准备进行推理测试...")
            print("模型加载成功，进行简单推理测试...")
            
            # 进行简单推理
            prompt = "什么是依赖管理？请简要解释。"
            outputs = llm.generate([prompt], sampling_params)
            
            logger.info("推理测试完成")
            print(f"\n推理结果:")
            print(outputs[0].outputs[0].text)
            
            print("\n" + "="*60)
            print("测试成功！原生vllm库功能正常。")
            print("建议: 可以考虑在应用中直接使用原生vllm库，绕过langchain_community的封装。")
            print("="*60)
            
        except Exception as model_error:
            logger.error(f"模型测试失败: {str(model_error)}", exc_info=True)
            print(f"\n模型测试失败: {str(model_error)}")
            print("\n可能的原因:")
            print("1. 网络问题导致无法下载gpt2模型")
            print("2. 内存不足")
            print("3. vllm与当前环境的兼容性问题")
            print("\n建议:")
            print("1. 确保网络连接正常")
            print("2. 尝试使用本地已下载的模型")
            print("3. 检查vllm的版本兼容性")
            
    except ImportError as import_error:
        logger.error(f"vllm导入失败: {str(import_error)}", exc_info=True)
        print(f"\nvllm导入失败: {str(import_error)}")
        print("\n可能的原因:")
        print("1. vllm包未正确安装")
        print("2. Python环境问题")
        print("3. vllm与当前Python版本不兼容")
        print("\n建议:")
        print("1. 重新安装vllm: pip install vllm --upgrade")
        print("2. 检查Python版本是否符合要求")
        print("3. 查看完整错误日志以获取更多信息")
        
    except Exception as e:
        logger.error(f"测试过程中发生未知错误: {str(e)}", exc_info=True)
        print(f"\n测试过程中发生未知错误: {str(e)}")


if __name__ == "__main__":
    main()