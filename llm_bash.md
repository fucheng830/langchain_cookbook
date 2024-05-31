```
# Bash链
这个笔记本展示了使用大型语言模型（LLM）和bash进程来执行简单的文件系统命令。


```python
from langchain_experimental.llm_bash.base import LLMBashChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)

text = "请编写一个bash脚本，将'Hello World'打印到控制台。"

bash_chain = LLMBashChain.from_llm(llm, verbose=True)

bash_chain.invoke(text)
```

    
    
    [1m> 进入新的LLMBashChain链...[0m
    请编写一个bash脚本，将'Hello World'打印到控制台。[32;1m[1;3m
    
    ```bash
    echo "Hello World"
    ```[0m
    代码：[33;1m[1;3m['echo "Hello World"'][0m
    答案：[33;1m[1;3mHello World
    [0m
    [1m> 完成链。[0m



## 自定义提示
你还可以自定义提示。以下是一个示例，提示避免使用'echo'实用程序


```python
from langchain.prompts.prompt import PromptTemplate
from langchain_experimental.llm_bash.prompt import BashOutputParser

_PROMPT_TEMPLATE = """如果有人让你执行一个任务，你的工作就是想出一系列bash命令来完成任务。不需要在答案中加上"#!/bin/bash"。确保一步一步地推理，使用这个格式：
问题："将目录名为'target'中的文件复制到与target同一级别的名为'myNewDirectory'的新目录中"
我需要采取以下行动：
- 列出目录中的所有文件
- 创建一个新的目录
- 将第一个目录中的文件复制到第二个目录中
```bash
ls
mkdir myNewDirectory
cp -r target/* myNewDirectory
```

不要使用'echo'来编写脚本。

那就是格式。开始！
问题：{question}"""

PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
    output_parser=BashOutputParser(),
)
```


```python
bash_chain = LLMBashChain.from_llm(llm, prompt=PROMPT, verbose=True)

text = "请编写一个bash脚本，将'Hello World'打印到控制台。"

bash_chain.invoke(text)
```

    
    
    [1m> 进入新的LLMBashChain链...[0m
    请编写一个bash脚本，将'Hello World'打印到控制台。[32;1m[1;3m
    
    ```bash
    printf "Hello World\n"
    ```[0m
    代码：[33;1m[1;3m['printf "Hello World\n"'][0m
    答案：[33;1m[1;3mHello World
    [0m
    [1m> 完成链。[0m



## 持久终端

默认情况下，每次调用链都会在单独的子进程中运行。可以通过实例化一个持久的bash进程来改变这种行为。


```python
from langchain_experimental.llm_bash.bash import BashProcess

persistent_process = BashProcess(persistent=True)
bash_chain = LLMBashChain.from_llm(llm, bash_process=persistent_process, verbose=True)

text = "列出当前目录然后向上移动一级。"

bash_chain.invoke(text)
```

    
    
    [1m> 进入新的LLMBashChain链...[0m
    列出当前目录然后向上移动一级。[32;1m[1;3m
    
    ```bash
    ls
    cd ..
    ```[0m
    代码