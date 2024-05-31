“
# 代理和工具之间的共享内存

这个笔记本介绍了如何在代理和其工具中添加内存。在阅读这个笔记本之前，请先阅读以下笔记本，因为这将建立在它们之上：

- [为LLM链添加内存](/docs/modules/memory/integrations/adding_memory)
- [自定义代理](/docs/modules/agents/how_to/custom_agent)

我们将创建一个自定义代理。该代理可以访问对话内存、搜索工具和总结工具。总结工具也需要访问对话内存。

```python
from langchain import hub
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent, create_react_agent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.prompts import PromptTemplate
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_openai import OpenAI
```

```python
template = """这是一个人类和机器人之间的对话：

{chat_history}

为{input}写一个对话摘要：
"""

prompt = PromptTemplate(input_variables=["input", "chat_history"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history")
readonlymemory = ReadOnlySharedMemory(memory=memory)
summary_chain = LLMChain(
    llm=OpenAI(),
    prompt=prompt,
    verbose=True,
    memory=readonlymemory  # 使用只读内存，防止工具修改内存
)
```

```python
search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="当你需要了解当前事件时，这个工具很有用",
    ),
    Tool(
        name="Summary",
        func=summary_chain.run,
        description="当你需要总结对话时，这个工具很有用。这个工具的输入应该是一个字符串，表示谁将阅读这个摘要。",
    ),
]
```

```python
prompt = hub.pull("hwchase17/react")
```

现在我们可以构造带有内存对象的`LLMChain`，然后创建代理。

```python
model = OpenAI()
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)
```

```python
agent_executor.invoke({"input": "ChatGPT是什么？"})
```

```python
agent_executor.invoke({"input": "谁开发了它？"})
```

```python
agent_executor.invoke({"input": "谢谢。为我5岁的女儿总结这次对话。"})
```

```python
print(agent_executor.memory.buffer)
```

为了测试这个代理的内存，我们可以问一个依赖于之前交换信息的后续问题，以确保正确回答。

```python
agent_executor.invoke({"input": "谁开发了它？"})
```

```python
agent_executor.invoke({"input": "谢谢。为我5岁的女儿总结这次对话。"})
```

```python
print(agent_executor.memory.buffer)
```

为了比较，下面是一个使用相同内存的代理和工具的坏例子。

```python
## 这是一个使用内存的坏做法。
## 使用ReadOnlySharedMemory类，如上面所示。

template = """这是一个人类和机器人之间的对话：

{chat_history}

为{input}写一个对话摘要：
"""

prompt = PromptTemplate(input_variables=["input", "chat_history"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history")
summary_chain = LLMChain(
    llm=OpenAI(),
    prompt=prompt,
    verbose=True,
    memory=memory  # <--- 唯一的变化
)

search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="当你需要了解当前事件时，这个工具很有用",
    