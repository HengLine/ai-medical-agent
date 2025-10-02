#!/usr/bin/env python
"""
医疗AI智能体API启动脚本

此脚本用于启动FastAPI服务，提供医疗智能体的RESTful API接口。
"""

import os
import sys

import uvicorn
from dotenv import load_dotenv

from hengline.logger import info, warning, error

# 加载.env文件中的环境变量
dotenv_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    info(f"==============已加载.env文件: {dotenv_path}")
else:
    warning(f"==============警告: 未找到.env文件，将使用系统环境变量")

# 确保项目根目录在Python路径中
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


# 确保项目根目录在Python路径中
def ensure_project_path():
    """确保项目根目录在Python路径中"""
    project_root = os.path.abspath(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.append(project_root)
        info(f"已将项目根目录添加到Python路径: {project_root}")


# 启动API服务
def start_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False, agent_type: str = None):
    """启动FastAPI服务"""
    # 使用项目的日志模块
    info(f"准备启动医疗AI智能体API服务...")
    info(f"服务将在 http://{host}:{port} 启动")
    info(f"API文档地址: http://{host}:{port}/docs")

    # 导入api_app并设置全局智能体类型
    try:
        from hengline.api import api_app
        api_app.set_global_agent_type(agent_type)
    except ImportError as e:
        error(f"无法导入api_app或设置全局智能体类型: {str(e)}")

    # 启动服务器
    try:
        uvicorn.run(
            "hengline.api.api_app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        info("用户中断，正在关闭服务...")
    except Exception as e:
        error(f"服务启动失败: {str(e)}")
        raise


if __name__ == "__main__":
    # 确保项目路径正确
    ensure_project_path()

    # 解析命令行参数
    import argparse

    parser = argparse.ArgumentParser(description='启动医疗AI智能体API服务')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务监听地址')
    parser.add_argument('--port', type=int, default=8000, help='服务监听端口')
    parser.add_argument('--reload', action='store_true', help='开发模式下启用自动重载')
    parser.add_argument(
        "--type",
        choices=["ollama", "vllm", "openai", "qwen"],
        default=None,
        help="选择智能体的后端类型 (默认: ollama)"
    )
    args = parser.parse_args()

    # 记录选择的智能体类型
    info(f"正在初始化医疗智能体...")

    # 启动服务，传递智能体类型参数
    start_api_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        agent_type=args.type
    )
