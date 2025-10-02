
import os
import sys

from typing import Optional, List, Union

from pydantic import BaseModel, Field, validator

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


# 请求和响应模型
class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str
    request_id: Optional[str] = None


class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str
    request_id: Optional[str] = None
    sources: Optional[str] = None
    timestamp: str


class LLMConfig(BaseModel):
    """LLM配置模型"""
    api_key: str = Field(default="")
    api_url: str = Field(default="http://localhost:11434")
    models: Union[str, List[str]] = Field(default="gemma3:4b")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    timeout: int = Field(default=300, ge=30)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1)
    repeat_penalty: float = Field(default=1.1, ge=0.0)

    @validator('models')
    def validate_models(cls, v):
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("模型列表不能为空")
        return v


class ConfigResponse(BaseModel):
    """配置响应模型"""
    llm: LLMConfig
    updated_at: str
    message: str = "配置获取成功"


class GenerationRequest(BaseModel):
    """生成式内容请求模型"""
    question: str
    type: str = Field(default="general_info",
                                description="生成类型",
                                pattern="^(general_info|detailed_explanation|patient_education|medical_case)$")
    request_id: Optional[str] = None


class GenerationResponse(BaseModel):
    """生成式内容响应模型"""
    answer: str
    type: str
    request_id: Optional[str] = None
    timestamp: str
