“
# 多智能体分散式发言人选择

这个笔记本展示了如何实现一个没有固定发言顺序的多智能体模拟。相反，智能体自己决定谁发言。我们可以通过让每个智能体出价来发言来实现这一点。出价最高的智能体可以发言。

我们将在下面的示例中展示如何实现这一点，该示例展示了一个虚构的总统辩论。

## 导入LangChain相关模块

```python
from typing import Callable, List

import tenacity
from langchain.output_parsers import RegexParser
from langchain.prompts import PromptTemplate
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
```

## `DialogueAgent`和`DialogueSimulator`类
我们将使用在[多玩家龙与地下城](https://python.langchain.com/en/latest/use_cases/agent_simulations/multi_player_dnd.html)中定义的相同的`DialogueAgent`和`DialogueSimulator`类。

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
        应用chatmodel到消息历史
        并返回消息字符串
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
        将{message} spoken by {name}连接到消息历史
        """
        self.message_history.append(f"{name}: {message}")


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
        以{message} from {name}开始对话
        """
        for agent in self.agents:
            agent.receive(name, message)

        # 增加时间
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. 选择下一个发言人
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. 下一个发言人发送消息
        message = speaker.send()

        # 3. 每个人都收到消息
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. 增加时间
        self._step += 1

        return speaker.name, message
```

## `BiddingDialogueAgent`类
我们定义了一个`DialogueAgent`的子类，该类有一个`bid()`方法，该方法根据消息历史和最近的回复产生一个出价。

```python
class BiddingDialogueAgent(DialogueAgent):
    def __init__(
        self,
        name,
        system_message: SystemMessage,
        bidding_template: PromptTemplate,
        model: ChatOpenAI,
    ) -> None:
        super().__init__(name, system_message, model)
        self.bidding_template = bidding_template

    def bid(self) -> str:
        """
        询问chatmodel输出一个发言出价