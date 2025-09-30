import requests
import json
import time

"""
医疗AI智能体 - 第三方API集成示例

此示例展示如何使用医疗智能体的外部API集成功能，通过调用第三方API执行智能体任务。
示例包括：
1. 执行单例医疗查询任务
2. 执行症状提取任务
3. 执行症状严重程度评估任务
4. 批量执行任务

注意：在实际使用前，请确保：
- 医疗智能体API服务正在运行
- 已配置有效的第三方API端点
"""

# API基础URL
BASE_URL = "http://localhost:8000/api/external"

# 第三方API配置示例
EXTERNAL_API_CONFIG = {
    "api_url": "https://api.example-medical-service.com/v1",  # 替换为实际的第三方API地址
    "api_key": "your-api-key-here",  # 如果需要API密钥
    "timeout": 30,  # 超时时间(秒)
    "headers": {}
}

class ExternalApiClient:
    """第三方API集成客户端"""
    
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        
    def execute_task(self, task_type, input_data, use_agent_to_process=True):
        """
        执行单个智能体任务
        
        参数:
            task_type: 任务类型，如'medical_query', 'symptom_extraction', 'severity_assessment'
            input_data: 任务输入数据
            use_agent_to_process: 是否使用医疗智能体处理第三方API结果
            
        返回:
            dict: 任务结果
        """
        url = f"{self.base_url}/execute-task"
        
        payload = {
            "external_api_config": EXTERNAL_API_CONFIG,
            "task_type": task_type,
            "input_data": input_data,
            "use_agent_to_process": use_agent_to_process
        }
        
        try:
            start_time = time.time()
            response = self.session.post(url, json=payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"任务执行成功 - 耗时: {response_time:.2f}秒")
                return result
            else:
                print(f"任务执行失败: HTTP {response.status_code}")
                print(f"错误详情: {response.text}")
                return {
                    "success": False,
                    "error_code": response.status_code,
                    "error_message": response.text
                }
                
        except requests.RequestException as e:
            print(f"请求异常: {str(e)}")
            return {
                "success": False,
                "error_message": str(e)
            }
            
    def batch_execute_tasks(self, tasks):
        """
        批量执行智能体任务
        
        参数:
            tasks: 任务列表，每个任务包含task_type和input_data
            
        返回:
            list: 任务结果列表
        """
        url = f"{self.base_url}/batch-execute-tasks"
        
        payload = {
            "external_api_config": EXTERNAL_API_CONFIG,
            "tasks": tasks
        }
        
        try:
            start_time = time.time()
            response = self.session.post(url, json=payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                results = response.json()
                print(f"批量任务执行完成 - 总耗时: {response_time:.2f}秒")
                return results
            else:
                print(f"批量任务执行失败: HTTP {response.status_code}")
                print(f"错误详情: {response.text}")
                return []
                
        except requests.RequestException as e:
            print(f"批量请求异常: {str(e)}")
            return []

# 使用示例
if __name__ == "__main__":
    client = ExternalApiClient()
    
    print("===== 医疗AI智能体 - 第三方API集成示例 =====")
    print("1. 执行医疗查询任务")
    
    # 1. 执行医疗查询任务
    print("\n=== 示例1: 执行医疗查询任务 ===")
    medical_query_result = client.execute_task(
        task_type="medical_query",
        input_data={
            "question": "高血压的预防措施有哪些？",
            "patient_info": {
                "age": 45,
                "gender": "male",
                "medical_history": ["轻度肥胖"]
            }
        },
        use_agent_to_process=True
    )
    print("查询结果:")
    print(json.dumps(medical_query_result, ensure_ascii=False, indent=2))
    
    # 2. 执行症状提取任务
    print("\n=== 示例2: 执行症状提取任务 ===")
    symptom_extraction_result = client.execute_task(
        task_type="symptom_extraction",
        input_data={
            "text": "患者主诉头痛三天，伴有恶心呕吐，体温38.5℃，血压145/90mmHg",
            "context": "门诊记录"
        },
        use_agent_to_process=True
    )
    print("症状提取结果:")
    print(json.dumps(symptom_extraction_result, ensure_ascii=False, indent=2))
    
    # 3. 执行症状严重程度评估任务
    print("\n=== 示例3: 执行症状严重程度评估任务 ===")
    severity_assessment_result = client.execute_task(
        task_type="severity_assessment",
        input_data={
            "symptoms": ["胸痛", "呼吸困难", "出汗", "左臂疼痛"],
            "patient_info": {
                "age": 62,
                "gender": "male",
                "medical_history": ["高血压", "冠心病"]
            }
        },
        use_agent_to_process=True
    )
    print("严重程度评估结果:")
    print(json.dumps(severity_assessment_result, ensure_ascii=False, indent=2))
    
    # 4. 批量执行任务
    print("\n=== 示例4: 批量执行任务 ===")
    batch_tasks = [
        {
            "task_type": "medical_query",
            "input_data": {
                "question": "糖尿病的早期症状有哪些？",
                "patient_info": {"age": 50}
            },
            "use_agent_to_process": True
        },
        {
            "task_type": "symptom_extraction",
            "input_data": {
                "text": "患者出现多饮、多尿、多食但体重减轻的症状",
                "context": "病历摘要"
            },
            "use_agent_to_process": True
        },
        {
            "task_type": "severity_assessment",
            "input_data": {
                "symptoms": ["高热", "意识模糊", "抽搐"],
                "patient_info": {"age": 5, "medical_history": ["无"]}
            },
            "use_agent_to_process": False
        }
    ]
    
    batch_results = client.batch_execute_tasks(batch_tasks)
    print(f"批量任务结果数量: {len(batch_results)}")
    
    for i, result in enumerate(batch_results):
        print(f"\n任务 {i+1} 结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n===== 示例执行完成 =====")
    print("\n提示:")
    print("1. 实际使用时，请替换EXTERNAL_API_CONFIG中的api_url为真实的第三方API地址")
    print("2. 如果第三方API需要认证，请填写有效的api_key")
    print("3. 可以根据需要调整任务类型和输入数据格式")
    print("4. use_agent_to_process参数决定是否使用医疗智能体进一步处理第三方API结果")