这是LangChain实现Meta-Prompt的描述，由Noah Goodman提出，用于构建自我改进的代理。

Meta-Prompt的核心思想是提示代理反思自己的表现并修改自己的指令。

代理是一个简单的循环，开始时没有任何指令，并遵循以下步骤：

与用户进行对话，用户可能提供请求、指令或反馈。

在会话结束时，使用元提示生成自我批评和新指令。

```
Assistant刚刚与用户进行了以下交互。Assistant严格遵循其“system: Instructions”。你的工作是批评Assistant的表现，然后修改指令，以便Assistant在未来能快速正确地响应。

####
{hist}
####

请反思这些交互。

首先，你应该批评Assistant的表现。Assistant能做得更好吗？Assistant应该记住关于这个用户的什么？这个用户总是想要什么？用“Critique: ...”表示。

接下来，你应该修改指令，以便Assistant在未来能快速正确地响应。Assistant的目标是在最少的交互中满足用户。Assistant只能看到新的指令，不能看到交互历史，所以任何重要的事情都必须在指令中总结。不要忘记当前指令中的任何重要细节！用“Instructions: ...”表示新的指令。
```

重复以上步骤。

这个系统（我称之为Meta-prompt）唯一的固定指令是管理代理指令修改的元提示。代理在会话之间没有记忆，除了每次修改的指令。尽管其简单性，这个代理可以随着时间的推移而学习并自我改进，通过将其指令中的有用细节融入其中。

## 设置
我们定义了两个链。一个作为“Assistant”，另一个作为“meta-chain”，它批评“Assistant”的表现并修改对“Assistant”的指令。

```python
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
```

```python
def initialize_chain(instructions, memory=None):
    if memory is None:
        memory = ConversationBufferWindowMemory()
        memory.ai_prefix = "Assistant"

    template = f"""
    Instructions: {instructions}
    {{{memory.memory_key}}}
    Human: {{human_input}}
    Assistant:"""

    prompt = PromptTemplate(
        input_variables=["history", "human_input"], template=template
    )

    chain = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(),
    )
    return chain


def initialize_meta_chain():
    meta_template = """
    Assistant刚刚与用户进行了以下交互。Assistant严格遵循其“Instructions”。你的工作是批评Assistant的表现，然后修改指令，以便Assistant在未来能快速正确地响应。

    ####

    {chat_history}

    ####

    请反思这些交互。

    首先，你应该批评Assistant的表现。Assistant能做得更好吗？Assistant应该记住关于这个用户的什么？这个用户总是想要什么？用“Critique: ...”表示。

    接下来，你应该修改指令，以便Assistant在未来能快速正确地响应。Assistant的目标是在最少的交互中满足用户。Assistant只能看到新的指令，不能看到交互历史，所以任何重要的事情都必须在指令中总结。不要忘记当前指令中的任何重要细节！用“Instructions: ...”表示新的指令。
    """

    meta_prompt = PromptTemplate(
        input_variables=["chat_history"], template=meta_template
    )

    meta_chain = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=meta_prompt,
        verbose=True,
    )
    return meta_chain


def get_chat_history(chain_memory):
    memory_key = chain_memory.memory_key
    chat_history = chain_memory.load_memory_variables(memory_key)[memory_key]
    return chat_history


def get_new_instructions(meta_output):
    delimiter = "Instructions: "
    new_instructions = meta_output[meta_output.find(delimiter) + len(delimiter) :]
