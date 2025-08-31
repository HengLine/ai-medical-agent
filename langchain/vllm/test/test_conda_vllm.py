import sys
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('vllm_test')

try:
    # 尝试导入vllm的核心模块
    import vllm
    logger.info(f"成功导入vllm，版本: {vllm.__version__}")
    
    # 尝试初始化一个简单的模型
    from vllm import LLM, SamplingParams
    
    # 使用轻量级模型进行测试
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
llm = LLM(
    model="gpt2",  # 使用HuggingFace的gpt2模型
    device="cpu",
    max_model_len=2048,
    trust_remote_code=True,
    dtype="float32",  # CPU模式下使用float32
    disable_log_requests=True
)
logger.info("成功初始化vllm模型")
    
    # 生成文本
    prompts = ["什么是依赖管理？请简要解释。"]
    outputs = llm.generate(prompts, sampling_params)
    
    # 打印结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"提示: {prompt}")
        print(f"生成: {generated_text}")
        
    logger.info("vllm测试成功!")
    sys.exit(0)
except ImportError as e:
    logger.error(f"导入vllm失败: {e}")
    print(f"错误: 导入vllm失败 - {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"vllm测试过程中出错: {e}")
    print(f"错误: vllm测试失败 - {e}")
    sys.exit(1)
