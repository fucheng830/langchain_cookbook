“# 多模态输出：图像与文本

本笔记本展示了如何使用非文本生成工具来创建多模态代理。

此示例仅限于文本和图像输出，并使用UUID来在工具和代理之间传递内容。

本示例使用Steamship来生成和存储生成的图像。默认情况下，生成的图像受权限保护。

您可以在此处获取您的Steamship API密钥：https://steamship.com/account/api

```python
import re

from IPython.display import Image, display
from steamship import Block, Steamship
```

```python
from langchain.agents import AgentType, initialize_agent
from langchain.tools import SteamshipImageGenerationTool
from langchain_openai import OpenAI
```

```python
llm = OpenAI(temperature=0)
```

## Dall-E

```python
tools = [SteamshipImageGenerationTool(model_name="dall-e")]
```

```python
mrkl = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
```

```python
output = mrkl.run("你如何想象一只鹦鹉踢足球？")
```

```python
def show_output(output):
    """展示代理的多模态输出。"""
    UUID_PATTERN = re.compile(
        r"([0-9A-Za-z]{8}-[0-9A-Za-z]{4}-[0-9A-Za-z]{4}-[0-9A-Za-z]{4}-[0-9A-Za-z]{12})"
    )

    outputs = UUID_PATTERN.split(output)
    outputs = [
        re.sub(r"^\W+", "", el) for el in outputs
    ]  # 清理前导和尾随的非单词字符

    for output in outputs:
        maybe_block_id = UUID_PATTERN.search(output)
        if maybe_block_id:
            display(Image(Block.get(Steamship(), _id=maybe_block_id.group()).raw()))
        else:
            print(output, end="\n\n")
```

## StableDiffusion

```python
tools = [SteamshipImageGenerationTool(model_name="stable-diffusion")]
```

```python
mrkl = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
```

```python
output = mrkl.run("你如何想象一只鹦鹉踢足球？")
```

”