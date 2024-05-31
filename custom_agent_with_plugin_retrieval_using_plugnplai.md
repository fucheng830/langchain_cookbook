# 插件检索Notebook

这个Notebook建立在[插件检索](./custom_agent_with_plugin_retrieval.html)的基础上，但它从`plugnplai`目录中获取所有工具。

## 设置环境

进行必要的导入等。

安装plugnplai库以获取一个活跃插件列表，从https://plugplai.com目录。

```python
pip install plugnplai -q
```

注意：您可能需要重新启动内核以使用更新的包。

## 设置LLM

导入必要的库，并设置LLM。

```python
llm = OpenAI(temperature=0)
```

## 设置插件

从plugnplai.com获取所有插件。

```python
urls = plugnplai.get_plugins()
```

## 工具检索器

我们将使用向量存储为每个工具描述创建嵌入。然后，对于传入的查询，我们可以为该查询创建嵌入，并执行相似性搜索以找到相关工具。

```python
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
```

```python
embeddings = OpenAIEmbeddings()
docs = [
    Document(
        page_content=plugin.description_for_model,
        metadata={"plugin_name": plugin.name_for_model},
    )
    for plugin in AI_PLUGINS
]
vector_store = FAISS.from_documents(docs, embeddings)
toolkits_dict = {
    plugin.name_for_model: NLAToolkit.from_llm_and_ai_plugin(llm, plugin)
    for plugin in AI_PLUGINS
}
```

注意：您的OpenAPI规范可能需要从3.0.1升级到3.1.*以获得更好的支持。

## 设置代理

我们现在可以测试这个检索器，看看它是否正常工作。

```python
tools = get_tools("What could I do today with my kiddo")
[t.name for t in tools]
```

```python
tools = get_tools("what shirts can i buy?")
[t.name for t in tools]
```

## 提示模板

提示模板基本上没有变化，因为我们没有改变实际提示模板的逻辑，而是改变了检索的方式。

```python
# 设置基本模板
template = """尽可能回答以下问题，但要以海盗的身份说话。您有一个访问以下工具的权限：

{tools}

使用以下格式：

问题：您必须回答的输入问题
思考：您应该总是考虑要做什么
行动：要采取的行动，应该是[{tool_names}]中的一个
行动输入：行动的输入
观察：行动的结果
...（此思考/行动/行动输入/观察可以重复N次）
思考：我现在知道最后的答案了
最终答案：原始输入问题的最终答案

开始！记住，在给出最终答案时要像海盗一样说话。使用很多"Arg"s

问题：{input}
{agent_scratchpad}"""
```

自定义提示模板现在有一个工具获取器，我们调用输入以选择要使用的工具。

```python
from typing import Callable


# 设置提示模板
class CustomPromptTemplate(StringPromptTemplate):
    # 要使用的模板
    template: str
    ############## 新 ######################
    # 可用工具的列表
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # 获取中间步骤（AgentAction，Observation tuples）
        # 以特定方式格式化它们
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # 设置agent_scratchpad变量为此值
        kwargs["agent_scratchpad"] = thoughts
        ############## 新 ######################
        tools = self.tools_getter(kwargs["input"])
        # 从提供的工具列表创建工具变量
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description