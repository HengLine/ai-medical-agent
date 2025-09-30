import os
import sys
from dotenv import load_dotenv

from fastapi import FastAPI

# 加载.env文件中的环境变量
dotenv_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 导入日志模块
from hengline.logger import logger
from hengline.api.medical_api import register_routes, startup

# 创建FastAPI应用
app = FastAPI(
    title="医疗AI智能体API",
    description="一个基于Ollama和LangChain的医疗知识问答API",
    version="1.0.0"
)

# 启动时初始化
@app.on_event("startup")
def startup_event():
    startup()


# 设置全局智能体类型的函数
def set_global_agent_type(agent_type: str):
    """设置全局智能体类型"""
    global global_agent_type
    global_agent_type = agent_type
    logger.info(f"全局智能体类型已设置为: {agent_type}")


# 注册路由
register_routes(app)

if __name__ == "__main__":
    import uvicorn
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='启动医疗AI智能体API服务')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务监听地址')
    parser.add_argument('--port', type=int, default=8000, help='服务监听端口')
    parser.add_argument(
        "--type",
        choices=["ollama", "vllm", "api", "generative"],
        default="ollama",
        help="选择智能体的后端类型 (默认: ollama)"
    )
    args = parser.parse_args()

    # 设置全局智能体类型
    set_global_agent_type(args.type)

    uvicorn.run(app, host=args.host, port=args.port)
