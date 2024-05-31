“# 自定义多动作代理

本笔记本将介绍如何创建您自己的自定义代理。

一个代理包含两个部分：

- 工具：代理可用的工具。
- 代理类本身：决定采取何种行动。

在本笔记本中，我们将逐步讲解如何创建一个能够一次性预测/采取多个步骤的自定义代理。

```python
from langchain.agents import AgentExecutor, BaseMultiActionAgent, Tool
from langchain_community.utilities import SerpAPIWrapper
```

```python
def 随机单词(查询: str) -> str:
    print("\n现在我正在做这个！")
    return "foo"
```

```python
搜索 = SerpAPIWrapper()
工具 = [
    Tool(
        name="搜索",
        func=搜索.run,
        description="当你需要回答有关当前事件的问题时很有用",
    ),
    Tool(
        name="随机单词",
        func=随机单词,
        description="调用此函数以获取一个随机单词。",
    ),
]
```

```python
from typing import Any, List, Tuple, Union

from langchain_core.agents import AgentAction, AgentFinish


class 假代理(BaseMultiActionAgent):
    """自定义假代理。"""

    @property
    def input_keys(self):
        return ["输入"]

    def 计划(
        self, 中间步骤: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        """根据输入决定要做什么。

        参数:
            中间步骤: LLM到目前为止所采取的步骤，以及观察结果
            **kwargs: 用户输入。

        返回:
            指定使用哪个工具的动作。
        """
        if len(中间步骤) == 0:
            return [
                AgentAction(工具="搜索", 工具输入=kwargs["输入"], 日志=""),
                AgentAction(工具="随机单词", 工具输入=kwargs["输入"], 日志=""),
            ]
        else:
            return AgentFinish(返回值={"输出": "bar"}, 日志="")

    async def 异步计划(
        self, 中间步骤: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        """根据输入决定要做什么。

        参数:
            中间步骤: LLM到目前为止所采取的步骤，以及观察结果
            **kwargs: 用户输入。

        返回:
            指定使用哪个工具的动作。
        """
        if len(中间步骤) == 0:
            return [
                AgentAction(工具="搜索", 工具输入=kwargs["输入"], 日志=""),
                AgentAction(工具="随机单词", 工具输入=kwargs["输入"], 日志=""),
            ]
        else:
            return AgentFinish(返回值={"输出": "bar"}, 日志="")
```

```python
代理 = 假代理()
```

```python
代理执行器 = AgentExecutor.from_agent_and_tools(
    代理=代理, 工具=工具, 详细=True
)
```

```python
代理执行器.run("截至2023年，加拿大有多少人口？")
```

```

```

```

```