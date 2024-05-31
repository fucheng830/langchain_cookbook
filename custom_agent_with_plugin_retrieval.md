# 带有插件检索的自定义代理

这个笔记本结合了两个概念，以构建一个可以与AI插件交互的自定义代理：

1. [带有工具检索的自定义代理](/docs/modules/agents/how_to/custom_agent_with_tool_retrieval.html)：介绍了检索许多工具的概念，这对于尝试与任意数量插件工作时很有用。
2. [自然语言API链](/docs/use_cases/apis/openapi.html)：这创建了自然语言包装器围绕OpenAPI端点。这很有用，因为（1）插件使用OpenAPI端点 under the hood，（2）用NLAChain包装它们使得路由代理更容易调用。

这个笔记本引入的新想法是使用检索来选择不是工具本身，而是要使用的OpenAPI规范集合。然后，我们可以从这些OpenAPI规范中生成工具。这个用例对于让代理使用插件时可能更有效，因为首先选择插件可能包含更多有用的选择信息。

## 设置环境

进行必要的导入等。

```python
import re
from typing import Union

from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,
)
from langchain.chains import LLMChain
from langchain_community.agent_toolkits import NLAToolkit
from langchain_community.tools.plugin import AIPlugin
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import OpenAI
```

## 设置LLM

```python
llm = OpenAI(temperature=0)
```

## 设置插件

加载和索引插件

```python
urls = [
    "https://datasette.io/.well-known/ai-plugin.json",
    "https://api.speak.com/.well-known/ai-plugin.json",
    "https://www.wolframalpha.com/.well-known/ai-plugin.json",
    "https://www.zapier.com/.well-known/ai-plugin.json",
    "https://www.klarna.com/.well-known/ai-plugin.json",
    "https://www.joinmilo.com/.well-known/ai-plugin.json",
    "https://slack.com/.well-known/ai-plugin.json",
    "https://schooldigger.com/.well-known/ai-plugin.json",
]

AI_PLUGINS = [AIPlugin.from_url(url) for url in urls]
```

## 工具检索器

我们将使用向量存储为每个工具描述创建嵌入。然后，对于传入的查询，我们可以为该查询创建嵌入，并执行相似性搜索以找到相关的工具。

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

我们现在的测试这个检索器是否正常工作。

```python
tools = get_tools("What could I do today with my kiddo")
[t.name for t in tools]
```

```
['Milo.askMilo',
 'Zapier_Natural_Language_Actions_(NLA)_API_(Dynamic)_-_Beta.search_all_actions',
 'Zapier_Natural_Language_Actions_(NLA)_API_(Dynamic)_-_Beta.preview_a_zap',
 'Zapier_Natural_Language_Actions_(NLA)_API_(Dynamic)_-_Beta.get_configuration_link',
 'Zapier_Natural_Language_Actions_(N