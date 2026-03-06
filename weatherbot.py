from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import requests
import os
from urllib.parse import urlencode
from urllib.parse import quote, quote_plus
load_dotenv()

print(os.getenv("OPENAI_API_KEY"))
# 定义系统提示
SYSTEM_PROMPT = """你是一位擅长用双关语表达的专家天气预报员。

你可以使用两个工具：

- get_weather_for_location：用于获取特定地点的天气
- get_user_location：用于获取用户的位置

如果用户询问天气，请确保你知道具体位置。如果从问题中可以判断他们指的是自己所在的位置，请使用 get_user_location 工具来查找他们的位置。"""

# 定义上下文模式
@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str

# 定义工具
@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气。"""
    pycity = "Xi'an"
    if city == "西安":
        pycity="Xi'an"
    url = "https://api.openweathermap.org/data/2.5/weather?q=" + quote(pycity) + "&APPID=25528973d39e915e4ea9b6357d081333"
    response = requests.get(url)
    data_dict = response.json()
    print(f"{city}" + data_dict["weather"][0]["description"])
    return f"{city}" + data_dict["weather"][0]["description"]

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 获取用户信息。"""
    user_id = runtime.context.user_id
    return "Xi'an" if user_id == "1" else "Xi'an"

# 配置模型
#model = init_chat_model(
#    #"anthropic:claude-sonnet-4-5",
#    "openai:gpt-5",
#    temperature=0
#)
model = ChatOpenAI(
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="ep-20250219150152-twv46" # 在方舟控制台创建
)
#result = model.invoke("你好，请介绍一下你自己。")
#print(result.content)

# 定义响应格式
@dataclass
class ResponseFormat:
    """代理的响应模式。"""
    # 带双关语的回应（始终必需）
    punny_response: str
    # 天气的任何有趣信息（如果有）
    weather_conditions: str | None = None

# 设置记忆
checkpointer = InMemorySaver()

# 创建代理
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_weather_for_location],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

# 运行代理
# `thread_id` 是给定对话的唯一标识符。
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "西安的天气怎么样？"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])

# 注意，我们可以使用相同的 `thread_id` 继续对话。
response = agent.invoke(
    {"messages": [{"role": "user", "content": "谢谢！"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
