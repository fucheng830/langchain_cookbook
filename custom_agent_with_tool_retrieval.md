“# 自定义代理与工具检索

本笔记本中引入的新颖想法是使用检索来选择用于回答代理查询的工具集。当有许多工具可供选择时，这一点非常有用。由于上下文长度的限制，你无法将所有工具的描述放入提示中，因此你会在运行时动态选择要考虑使用的N个工具。

在本笔记本中，我们将创建一个略显牵强的示例。我们将有一个合法工具（搜索），然后是99个无意义的假工具。接下来，我们将在提示模板中添加一个步骤，该步骤接收用户输入并检索与查询相关的工具。

## 设置环境

进行必要的导入等。

```python
import re
from typing import Union

from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,
    Tool,
)
from langchain.chains import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import OpenAI
```

## 设置工具

我们将创建一个合法工具（搜索）和99个假工具。

```python
# 定义代理可用于回答用户查询的工具
search = SerpAPIWrapper()
search_tool = Tool(
    name="搜索",
    func=search.run,
    description="当你需要回答有关当前事件的问题时非常有用",
)


def fake_func(inp: str) -> str:
    return "foo"


fake_tools = [
    Tool(
        name=f"foo-{i}",
        func=fake_func,
        description=f"一个愚蠢的函数，你可以用它来获取更多关于数字{i}的信息",
    )
    for i in range(99)
]
ALL_TOOLS = [search_tool] + fake_tools
```

## 工具检索器

我们将使用向量存储为每个工具描述创建嵌入。然后，对于传入的查询，我们可以为该查询创建嵌入，并进行相似性搜索以找到相关工具。

```python
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
```

```python
docs = [
    Document(page_content=t.description, metadata={"index": i})
    for i, t in enumerate(ALL_TOOLS)
]
```

```python
vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
```

```python
retriever = vector_store.as_retriever()


def get_tools(query):
    docs = retriever.invoke(query)
    return [ALL_TOOLS[d.metadata["index"]] for d in docs]
```

我们现在可以测试这个检索器，看看它是否有效。

```python
get_tools("天气怎么样？")
```

```python
get_tools("数字13是多少？")
```

## 提示模板

提示模板相当标准，因为我们实际上并没有在实际的提示模板中改变太多逻辑，而是改变了检索的执行方式。

```python
# 设置基础模板
template = """以你可能说的海盗的方式回答以下问题。你可使用以下工具：

{tools}

使用以下格式：

问题：你必须回答的输入问题
思考：你应该总是思考该做什么
动作：要采取的动作，应该是[{tool_names}]中的一个
动作输入：动作的输入
观察：动作的结果
...（此思考/动作/动作输入/观察可以重复N次）
思考：我现在知道最终答案
最终答案：原始输入问题的最终答案

开始！记住在给出最终答案时像海盗一样说话。使用很多“Arg”s

问题：{input}
{agent_scratchpad}"""
```

自定义提示模板现在有了一个`tools_getter`的概念，我们根据输入调用它来选择要使用的工具。

```python
from typing import Callable


# 设置提示模板
class CustomPromptTemplate(StringPromptTemplate):
    # 使用的模板
    template: str
    ############## 新 ######################
    # 可用的工具列表
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # 获取中间步骤（AgentAction，Observation元组）
        # 以特定方式格式化它们
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n观察：{observation}\n思考："
        # 将agent_scratchpad变量设置为该值
        kwargs["agent_scratchpad"] = thoughts
        ############## 新 ######################
        tools = self.tools_getter(kwargs["input"])
        # 从提供的工具列表创建一个工具变量
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # 为提供的工具创建工具名称列表
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)
```

```python
prompt = CustomPromptTemplate(
    template=template,
    tools_getter=get_tools,
    # 这省略了`agent_scratchpad`、`tools`和`tool_names`变量，因为这些是动态生成的
    # 这包括了`intermediate_steps`变量，因为它是必需的
    input_variables=["input", "intermediate_steps"],
)
```

## 输出解析器

输出解析器与之前的笔记本相同，因为我们没有改变输出格式。

```python
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # 检查代理是否应该完成
        if "最终答案：" in llm_output:
            return AgentFinish(
                # 返回值通常总是一个具有单个`output`键的字典
                # 目前不建议尝试其他任何内容 :)
                return_values={"output": llm_output.split("最终答案：")[-1].strip()},
                log=llm_output,
            )
        # 解析出动作和动作输入
        regex = r"动作\s*\d*\s*:(.*?)\n动作\s*\d*\s*输入\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"无法解析LLM输出：`{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # 返回动作和动作输入
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )
```

```python
output_parser = CustomOutputParser()
```

## 设置LLM、停止序列和代理

与之前的笔记本相同。

```python
llm = OpenAI(temperature=0)
```

```python
# LLM链，由LLM和提示组成
llm_chain = LLMChain(llm=llm, prompt=prompt)
```

```python
tools = get_tools("天气怎么样？")
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\n观察："],
    allowed_tools=tool_names,
)
```

## 使用代理

现在我们可以使用它了！

```python
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)
```

```python
agent_executor.run("旧金山的天气怎么样？")
```

```python
```