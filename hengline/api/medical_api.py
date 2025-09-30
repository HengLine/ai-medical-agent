import json
import os
import sys
from datetime import datetime

from fastapi import FastAPI, HTTPException

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 导入日志模块
from hengline.logger import logger

# 导入配置读取器和智能体工厂
from hengline.config import ConfigReader
from hengline.agent.medical_agent import MedicalAgentFactory
from hengline.api.medical_model import QueryRequest, QueryResponse, LLMConfig, ConfigResponse, GenerationRequest, GenerationResponse

# 初始化配置读取器和医疗智能体
config_reader = ConfigReader()
medical_agent = None
generative_agent = None
# 存储从命令行传递的智能体类型
global_agent_type = None


def startup():
    """应用启动时初始化医疗智能体"""
    global medical_agent, generative_agent
    try:
        logger.info("正在初始化医疗智能体...")
        # 确定使用的智能体类型
        agent_type = global_agent_type if global_agent_type else config_reader.get_all_config().get("default_llm", "ollama")
        logger.info(f"使用 {agent_type} 类型的智能体")
        # 使用工厂创建相应类型的智能体
        medical_agent, generative_agent = MedicalAgentFactory.create_agent(agent_type)
        logger.info("医疗智能体初始化成功")

        # 对于生成式智能体，额外记录支持的功能
        logger.info("生成式智能体支持多种内容生成模式：general_info, detailed_explanation, patient_education, medical_case")
    except Exception as e:
        logger.error(f"医疗智能体初始化失败: {str(e)}")
        # 即使初始化失败，API仍会启动，但调用时会返回错误


# API路由
def register_routes(app: FastAPI):
    """注册所有API路由"""

    @app.get("/api/health", summary="健康检查", description="检查API和智能体的健康状态")
    def health_check():
        """检查API和智能体的健康状态"""
        status = {
            "api_status": "running",
            "agent_status": "initialized" if medical_agent is not None else "not initialized"
        }
        return status

    @app.put("/api/config", response_model=ConfigResponse, summary="更新LLM配置", description="更新LLM的配置信息")
    def update_config(config: LLMConfig):
        """更新LLM的配置信息"""
        try:
            # 声明全局变量
            global config_reader
            global medical_agent

            # 从配置中读取默认的LLM类型
            default_llm = config_reader.get_value("default_llm", "ollama")

            # 获取当前完整配置
            all_config = config_reader.get_all_config()

            # 更新对应类型LLM的配置
            if default_llm == "ollama":
                all_config["llm"][default_llm]["base_url"] = config.api_url
                all_config["llm"][default_llm]["model_name"] = config.models
            elif default_llm == "vllm":
                all_config["llm"][default_llm]["model"] = config.models
            else:
                all_config["llm"][default_llm]["api_url"] = config.api_url
                all_config["llm"][default_llm]["model"] = config.models

            # 更新通用配置
            all_config["llm"][default_llm]["temperature"] = config.temperature
            all_config["llm"][default_llm]["timeout"] = config.timeout
            all_config["llm"][default_llm]["top_p"] = config.top_p
            all_config["llm"][default_llm]["top_k"] = config.top_k

            # 如果有API密钥，更新API密钥
            if "api_key" in all_config["llm"][default_llm]:
                all_config["llm"][default_llm]["api_key"] = config.api_key

            # 写入配置文件
            with open(config_reader.config_path, 'w', encoding='utf-8') as f:
                json.dump(all_config, f, ensure_ascii=False, indent=4)

            # 重新初始化配置读取器
            config_reader = ConfigReader()

            # 重新初始化医疗智能体
            try:
                logger.info("更新配置后，重新初始化医疗智能体...")
                medical_agent = MedicalAgentFactory.create_agent(default_llm)
                logger.info("医疗智能体重新初始化成功")
            except Exception as e:
                logger.error(f"医疗智能体重新初始化失败: {str(e)}")
                # 不抛出异常，配置已成功更新，只是智能体重初始化失败

            return ConfigResponse(
                llm=config,
                updated_at=datetime.now().isoformat(),
                message=f"{default_llm} 类型的智能体配置更新成功"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"更新配置文件时发生错误: {str(e)}")

    @app.post("/api/query", response_model=QueryResponse, summary="查询医疗智能体", description="向医疗智能体发送问题并获取回答")
    def query_agent(request: QueryRequest):
        """向医疗智能体发送问题并获取回答"""
        if medical_agent is None:
            raise HTTPException(status_code=503, detail="医疗智能体未初始化，请稍后再试")

        if not request.question or request.question.strip() == "":
            raise HTTPException(status_code=400, detail="问题不能为空")

        try:
            # 调用医疗智能体回答问题
            result = medical_agent.run(request.question)

            # 构建响应
            response = QueryResponse(
                answer=result,
                request_id=request.request_id,
                timestamp=datetime.now().isoformat()
            )

            # 尝试提取来源信息（如果有）
            if isinstance(result, str) and "来源:" in result:
                parts = result.split("来源:")
                response.answer = parts[0].strip()
                response.sources = parts[1].strip() if len(parts) > 1 else None

            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")

    @app.post("/api/generate", response_model=GenerationResponse, summary="生成医疗内容", description="生成指定主题的医疗内容")
    def generate_content(request: GenerationRequest):
        """生成指定主题的医疗内容"""
        if generative_agent is None:
            raise HTTPException(status_code=503, detail="医疗智能体未初始化，请稍后再试")

        if not request.topic or request.topic.strip() == "":
            raise HTTPException(status_code=400, detail="主题不能为空")

        try:
            # 检查是否是生成式智能体
            if not hasattr(generative_agent, "generate_content"):
                # 获取当前使用的智能体类型
                agent_type = config_reader.get_all_config().get("default_llm", "ollama")
                raise HTTPException(status_code=400, 
                                   detail=f"当前使用的是{agent_type}类型智能体，不支持生成式功能。请切换到generative类型智能体。")

            # 调用生成式智能体生成内容
            result = generative_agent.generate_content(
                topic=request.topic,
                generation_type=request.generation_type
            )

            # 构建响应
            response = GenerationResponse(
                generated_content=result,
                generation_type=request.generation_type,
                request_id=request.request_id,
                timestamp=datetime.now().isoformat()
            )

            return response
        except HTTPException as e:
            # 重新抛出HTTP异常
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"生成内容时发生错误: {str(e)}")
