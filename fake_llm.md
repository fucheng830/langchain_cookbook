“# 假LLM
LangChain 提供了一个假的 LLM 类，用于测试。这允许你模拟对 LLM 的调用，并模拟如果 LLM 以某种方式响应会发生什么。

在本笔记本中，我们将介绍如何使用这个假 LLM。

我们从使用 FakeLLM 在一个代理中开始。

```python
from langchain_community.llms.fake import FakeListLLM
```

```python
from langchain.agents import AgentType, initialize_agent, load_tools
```

```python
tools = load_tools(["python_repl"])
```

```python
responses = ["Action: Python REPL\nAction Input: print(2 + 2)", "Final Answer: 4"]
llm = FakeListLLM(responses=responses)
```

```python
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
```

```python
agent.invoke("2 + 2 是多少")
```

    
    
    [1m> 进入新的 AgentExecutor 链...[0m
    [32;1m[1;3m动作：Python REPL
    动作输入：print(2 + 2)[0m
    观察： [36;1m[1;3m4
    [0m
    思考：[32;1m[1;3m最终答案：4[0m
    
    [1m> 链结束。[0m





    '4'




```python

```

”