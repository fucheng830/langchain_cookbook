“
# 多玩家地下城与龙

这个笔记本展示了如何使用 `DialogueAgent` 和 `DialogueSimulator` 类轻松地将 [两人地下城与龙示例](https://python.langchain.com/en/latest/use_cases/agent_simulations/two_player_dnd.html) 扩展到多个玩家。

模拟两个玩家和多个玩家的主要区别在于每个玩家发言的调度。

为此，我们增强 `DialogueSimulator` 以接受一个自定义函数，该函数确定哪个玩家发言。在下面的示例中，每个角色都以轮流的方式发言，故事讲述者在每个玩家之间交替。

## 导入与LangChain相关的模块

```python
from typing import Callable, List

from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
```

## `DialogueAgent` 类
`DialogueAgent` 类是一个简单的 `ChatOpenAI` 模型包装器，通过简单地将消息历史串联起来，从 `dialogue_agent` 的角度存储消息历史。

它公开了两个方法：
- `send()`：将消息历史应用于聊天模型，并返回消息字符串
- `receive(name, message)`：将 `message` 添加到消息历史中，该 `message` 由 `name` 说出

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

## `DialogueSimulator` 类
`DialogueSimulator` 类接受一个玩家列表。在每一步，它执行以下操作：
1. 选择下一个发言人
2. 调用下一个发言人的 `send()` 方法
3. 将消息广播给所有其他玩家
4. 更新步骤计数器。
选择下一个发言人的方式可以实现为任何函数，但在此情况下我们简单地遍历玩家。

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
        self._step += 