以下是您提供的文本的中文翻译：

---

# 探索 OpenAI V1 功能

在 2023 年 6 月 11 日，OpenAI 发布了一系列新功能，并将其 Python SDK 更新到了 1.0.0 版本。这个笔记本展示了新功能以及如何使用它们与 LangChain 结合。

```python
# 需要 openai>=1.1.0, langchain>=0.0.335, langchain-experimental>=0.0.39
!pip install -U openai langchain langchain-experimental
```

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
```

## [视觉](https://platform.openai.com/docs/guides/vision)

OpenAI 发布了多模态模型，可以接受文本和图像序列作为输入。

```python
chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=256)
chat.invoke(
    [
        HumanMessage(
            content=[
                {"type": "text", "text": "这张图片展示了什么"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/langchain_stack.png",
                        "detail": "auto",
                    },
                },
            ]
        )
    ]
)
```

## [OpenAI 助手](https://platform.openai.com/docs/assistants/overview)

> Assistant API 允许您在自己的应用程序中构建 AI 助手。一个助手有指令并且可以利用模型、工具和知识来响应用户查询。当前 Assistant API 支持三种类型的工具：代码解释器、检索器和函数调用。

当使用仅 OpenAI 工具时，您可以直接调用助手并获取最终答案。当使用自定义工具时，您可以使用内置的 AgentExecutor 运行助手和工具执行循环，或者轻松编写自己的执行器。

以下是如何与助手交互的不同方式。作为一个简单的例子，让我们构建一个可以写和运行代码的数学导师。

### 使用仅 OpenAI 工具

```python
from langchain.agents.openai_assistant import OpenAIAssistantRunnable
```

```python
interpreter_assistant = OpenAIAssistantRunnable.create_assistant(
    name="langchain assistant",
    instructions="你是一个个人数学导师。写和运行代码来回答数学问题。",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview",
)
output = interpreter_assistant.invoke({"content": "10 - 4 的 2.7 次方是多少"})
output
```

### 作为 LangChain 代理使用任意工具

现在让我们使用自己的工具来重现这个功能。作为一个例子，我们将使用 [E2B 沙盒运行时工具](https://e2b.dev/docs?ref=landing-page-get-started)。

```python
!pip install e2b duckduckgo-search
```

```python
from langchain.tools import DuckDuckGoSearchRun, E2BDataAnalysisTool

tools = [E2BDataAnalysisTool(api_key="..."), DuckDuckGoSearchRun()]
```

```python
agent = OpenAIAssistantRunnable.create_assistant(
    name="langchain assistant e2b tool",
    instructions="你是一个个人数学导师。写和运行代码来回答数学问题。你也可以搜索互联网。",
    tools=tools,
    model="gpt-4-1106-preview",
    as_agent=True,
)
```

#### 使用 AgentExecutor

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools)
agent_executor.invoke({"content": "今天的 SF 天气除以 2.7 是什么"})
```

####