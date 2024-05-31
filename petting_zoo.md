“
# 多智能体模拟环境：宠物动物园

在这个例子中，我们展示了如何定义具有模拟环境的多个智能体模拟。像我们之前[使用Gymnasium的单智能体示例](https://python.langchain.com/en/latest/use_cases/agent_simulations/gymnasium.html)一样，我们创建了一个外部定义环境的智能体-环境循环。主要区别在于我们现在使用多个智能体而不是一个来实现这种交互循环。我们将使用[Petting Zoo](https://pettingzoo.farama.org/)库，这是[Gymnasium](https://gymnasium.farama.org/)的多智能体对应物。

## 安装 `pettingzoo` 和其他依赖项


```python
!pip install pettingzoo pygame rlcard
```

## 导入模块


```python
import collections
import inspect

import tenacity
from langchain.output_parsers import RegexParser
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
```

## `GymnasiumAgent`
这里我们复制了[我们的Gymnasium示例](https://python.langchain.com/en/latest/use_cases/agent_simulations/gymnasium.html)中定义的相同的`GymnasiumAgent`。如果在多次重试后它没有采取有效的行动，它只是采取随机行动。


```python
class GymnasiumAgent:
    @classmethod
    def get_docs(cls, env):
        return env.unwrapped.__doc__

    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.docs = self.get_docs(env)

        self.instructions = """
你的目标是最大化你的回报，即你收到的奖励的总和。
我会给你一个观察结果，奖励，终止标志，截断标志和到目前为止的回报，格式为：

观察结果：<observation>
奖励：<reward>
终止：<termination>
截断：<truncation>
回报：<sum_of_rewards>

你将回复一个行动，格式为：

行动：<action>

你用实际的<action>替换<action>。
除了返回行动外，不要做其他事情。
"""
        self.action_parser = RegexParser(
            regex=r"Action: (.*)", output_keys=["action"], default_output_key="action"
        )

        self.message_history = []
        self.ret = 0

    def random_action(self):
        action = self.env.action_space.sample()
        return action

    def reset(self):
        self.message_history = [
            SystemMessage(content=self.docs),
            SystemMessage(content=self.instructions),
        ]

    def observe(self, obs, rew=0, term=False, trunc=False, info=None):
        self.ret += rew

        obs_message = f"""
观察结果：{obs}
奖励：{rew}
终止：{term}
截断：{trunc}
回报：{self.ret}
        """
        self.message_history.append(HumanMessage(content=obs_message))
        return obs_message

    def _act(self):
        act_message = self.model.invoke(self.message_history)
        self.message_history.append(act_message)
        action = int(self.action_parser.parse(act_message.content)["action"])
        return action

    def act(self):
        try:
            for attempt in tenacity.Retrying(
                stop=tenacity.stop_after_attempt(2),
                wait=tenacity.wait_none(),  # 重试之间没有等待时间
                retry=tenacity.retry_if_exception_type(ValueError),
                before_sleep=lambda retry_state: print(
                    f"ValueError发生：{retry_state.outcome.exception()}, 正在重试..."
                ),
            ):
                with attempt:
                    action = self._act()
        except tenacity.RetryError:
