以下是您提供的内容的中文翻译：

```
# 两人玩家龙与地下城

在本手册中，我们将展示如何使用来自[CAMEL](https://www.camel-ai.org/)的概念来模拟一个角色扮演游戏，其中有一个主角和一位地下城主。我们创建了一个`DialogueSimulator`类来协调两个代理之间的对话。

## 导入LangChain相关模块

```python
from typing import Callable, List

from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
```

## `DialogueAgent`类
`DialogueAgent`类是`ChatOpenAI`模型的简单包装，它通过简单地连接消息来存储`dialogue_agent`的观点下的消息历史。

它暴露了两个方法：
- `send()`：将聊天模型应用于消息历史并返回消息字符串
- `receive(name, message)`：将`message` spoken by `name`添加到消息历史中

```python
class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model.invoke(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")
```

## `DialogueSimulator`类
`DialogueSimulator`类接受一个代理列表。在每一步，它执行以下操作：
1. 选择下一个发言者
2. 调用下一个发言者发送消息
3. 将消息广播给所有其他代理
4. 更新步骤计数器。
下一个发言者的选择可以实现为任何函数，但在这个例子中，我们简单地遍历代理。

```python
class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        message = speaker.send()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1

        return speaker.name, message
```

## 定义角色和任务

```python
protagonist_name = "哈利·波特"
storyteller_name = "地下城主"
quest = "找到伏地魔的七个魂器。"
word_limit = 50  # 任务头脑风暴的字数限制
``