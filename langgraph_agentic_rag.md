以下是将上述内容翻译成中文的版本：

```plaintext
# LangGraph 检索代理

我们可以根据[检索代理](https://python.langchain.com/docs/use_cases/question_answering/conversational_retrieval_agents)在[LangGraph](https://python.langchain.com/docs/langgraph)中实现。

## 检索器

### 初始化

我们从指定的URLs加载文档，并使用[RecursiveCharacterTextSplitter](https://python-langchain.readthedocs.io/en/latest/api/text_splitter.html#RecursiveCharacterTextSplitter)将文档分割成更小的部分。然后，我们将这些部分添加到向量数据库中。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 指定URLs
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 加载文档
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# 分割文档
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# 向量数据库添加文档
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()
```

### 检索代理

接下来，我们创建一个检索代理，该代理使用向量数据库来检索与用户查询相关的信息。

```python
from langchain.tools.retriever import create_retriever_tool

tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "搜索并返回关于Lilian Weng博客文章的信息。"
)

tools = [tool]

from langgraph.prebuilt import ToolExecutor

tool_executor = ToolExecutor(tools)
```

## 代理状态

我们定义一个图来表示代理的状态。每个节点都可能是一个函数或一个可执行的实体。

```python
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
```

## 节点和边

每个节点都可能是一个函数或一个可执行的实体。边则负责决定下一个节点。

```python
import json
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolInvocation

### 边

def should_retrieve(state):
    """
    决定代理是否应该检索更多信息或结束过程。

    此函数检查状态中的最后一个消息是否包含一个函数调用。如果存在函数调用，则继续检索信息。否则，结束过程。
    """
    # 省略实现代码

def check_relevance(state):
    """
    根据检索到的文档的相关性，决定代理的行为。

    此函数检查最后一个消息是否为FunctionMessage，表明已经执行了文档检索。然后，它使用预定义的模型和输出解析