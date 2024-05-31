以下是您提供的文本的中文翻译：

```
# BabyAGI with Tools

这个笔记本在[baby agi](baby_agi.html)的基础上，展示了如何更换执行链。之前的执行链只是一个人工智能，它会随机生成内容。通过用具有工具的智能体替换执行链，我们希望能够获得真实可靠的信息。

## 安装和导入所需模块

```python
from typing import Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_experimental.autonomous_agents import BabyAGI
from langchain_openai import OpenAI, OpenAIEmbeddings
```

## 连接到向量存储

根据您使用的向量存储，这一步可能会有所不同。

```python
%pip install faiss-cpu > /dev/null
%pip install google-search-results > /dev/null
from langchain.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
```

注意：您可能需要重新启动内核以使用更新的包。

## 定义链

BabyAGI依赖于三个LLM链：
- 任务创建链，用于选择要添加到列表的新任务
- 任务优先级链，用于重新优先排序任务
- 执行链，用于执行任务

注意：在此笔记本中，执行链将是一个智能体。

```python
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import OpenAI

todo_prompt = PromptTemplate.from_template(
    "你是 一个规划者，擅长为给定目标制定待办事项列表。为这个目标制定待办事项列表：{objective}"
)
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="当需要回答关于当前事件的问题时，非常有用"
    ),
    Tool(
        name="TODO",
        func=todo_chain.run,
        description="当需要制定待办事项列表时非常有用。输入：一个需要制定待办事项列表的目标。输出：该目标的待办事项列表。请清楚地说明目标是什么！"
    ),
]

```

### 运行BabyAGI

现在，是时候创建BabyAGI控制器并观察它试图实现您的目标了。

```python
OBJECTIVE = "为SF今天写一份天气报告"
```

```python
# 日志记录LLMChains
verbose = False
# 如果为None，将无限期继续
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    task_execution_chain=agent_executor,
    verbose=verbose,
    max_iterations=max_iterations,
)
```

```python
baby_agi({"objective": OBJECTIVE})
```

```
*****TASK LIST*****
1: 制定待办事项列表
*****NEXT TASK*****
1: 制定待办事项列表


> 进入新的AgentExecutor链...
Thought: 我需要制定一个待办事项列表
Action: TODO
Action Input: 为SF今天写一份天气报告
```

1. 研究旧金山当前的天气条件
2. 收集有关温度、湿度、风速和其他相关天气条件的数据
3. 分析数据以确定当前天气趋势
4. 写一个简短的介绍性段落
5. 描述旧金山当前的天气条件
6. 讨论任何即将到来的天气变化
7. 总结天气报告
8. 校对和编辑报告
9. 提交报告

```
> 完成链。
*****TASK RESULT*****
为SF今天写一份天气报告的待办事项列表是：1. 研究旧金山当前的天气条件；2. 收集有关温度、湿度、风速和其他相关天气