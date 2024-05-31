```python
! pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain langgraph

# 自RAG

自RAG是最近的一篇论文，它介绍了一种有趣的主动RAG方法。

该框架训练了一个单一的任意语言模型（LLaMA2-7b, 13b）来生成控制RAG过程的令牌：

1. 我应该从检索器中检索吗，`R` -

* 令牌：`Retrieve`
* 输入：`x (问题)` 或 `x (问题)`，`y (生成)`
* 决定何时从`D`中检索`R`片段
* 输出：`是，否，继续`

2. 检索到的片段`D`与问题`x`相关吗 -

* 令牌：`ISREL`
* * 输入：(`x (问题)`，`d (片段)`) 对于 `d` 在 `D` 中
* `d` 为解决问题`x`提供了有用的信息
* 输出：`相关，不相关`

3. 每个`D`片段中的LLM生成是否与片段相关（幻觉等） -

* 令牌：`ISSUP`
* 输入：`x (问题)`，`d (片段)`，  `y (生成)` 对于 `d` 在 `D` 中
* `y (生成)`中的所有可验证的陈述都由`d`支持
* 输出：`{完全支持，部分支持，无支持}`

4. 每个`D`片段中的LLM生成是否是对`x (问题)`的有用响应 -

* 令牌：`ISUSE`
* 输入：`x (问题)`，`y (生成)` 对于 `d` 在 `D` 中
* `y (生成)`是对`x (问题)`的有用响应。
* 输出：`{5, 4, 3, 2, 1}`

我们可以将此表示为图形：

![屏幕截图 2024-02-02 在 1.36.44 PM.png](ea6a57d2-f2ec-4061-840a-98deb3207248.png)

论文 -

https://arxiv.org/abs/2310.11511

---

让我们使用[LangGraph](https://python.langchain.com/docs/langgraph)从头实现这个。

## 检索器

让我们索引3篇博客文章。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# 添加到向量数据库
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()
```

## 状态

我们将定义一个图形。

我们的状态将是一个`dict`。

我们可以从任何图形节点访问此内容，如`state['keys']`。

```python
from typing import Dict, TypedDict

