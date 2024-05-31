“# Nomic嵌入模型

Nomic发布了一款新的嵌入模型，该模型在长上下文检索方面表现出色（8k上下文窗口）。

本食谱将指导您如何使用Nomic嵌入模型构建并部署（通过LangServe）一个RAG应用。

![截图 2024-02-01 上午9.14.15.png](4015a2e2-3400-4539-bd93-0d987ec5a44e.png)

## 注册

获取您的API令牌，然后运行：
```
! nomic 登录
```

然后使用生成的API令牌运行：
```
! nomic 登录 < 令牌 >
```

```python
! nomic 登录
```

```python
! nomic 登录 令牌
```

```python
! pip install -U langchain-nomic langchain_community tiktoken langchain-openai chromadb langchain
```

```python
# 可选：LangSmith API密钥
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "api_key"
```

## 文档加载

我们来测试三篇有趣的博客文章。

```python
from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
```

## 分割

长上下文检索

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=7500, chunk_overlap=100
)
doc_splits = text_splitter.split_documents(docs_list)
```

```python
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
for d in doc_splits:
    print("文档包含 %s 个令牌" % len(encoding.encode(d.page_content)))
```

## 索引

Nomic嵌入模型 [此处](https://docs.nomic.ai/reference/endpoints/nomic-embed-text)。

```python
import os

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_nomic import NomicEmbeddings
from langchain_nomic.embeddings import NomicEmbeddings
```

```python
# 添加到向量数据库
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=NomicEmbeddings(model="nomic-embed-text-v1"),
)
retriever = vectorstore.as_retriever()
```

## RAG链

我们可以使用Mistral `v0.2`，它是[针对32k上下文进行微调的](https://x.com/dchaplot/status/1734198245067243629?s=20)。

我们可以[使用Ollama](https://ollama.ai/library/mistral) -
```
ollama pull mistral:instruct
```

我们也可以运行[GPT-4 128k](https://openai.com/blog/new-models-and-developer-products-announced-at-devday)。

```python
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 提示
template = """仅基于以下上下文回答问题：
{context}

问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM API
model = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

# 本地LLM
ollama_llm = "mistral:instruct"
model_local = ChatOllama(model=ollama_llm)

# 链
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model_local
    | StrOutputParser()
)
```

```python
# 问题
chain.invoke("代理记忆的类型有哪些？")
```

**Mistral**

跟踪：24k提示令牌。

* https://smith.langchain.com/public/3e04d475-ea08-4ee3-ae66-6416a93d8b08/r

---

在[针在干草堆分析](https://twitter.com/GregKamradt/status/1722386725635580292?lang=en)中注意到一些考虑因素：

* LLMs可能在从大型上下文中检索信息时遇到困难，这取决于信息放置的位置。

## LangServe

创建一个LangServe应用。

![截图 2024-02-01 上午10.36.05.png](0afd4ea4-7ba2-4bfb-8e6d-57300e7a651f.png)

```
$ conda create -n template-testing-env python=3.11
$ conda activate template-testing-env
$ pip install -U "langchain-cli[serve]" "langserve[all]"
$ langchain app new .
$ poetry add langchain-nomic langchain_community tiktoken langchain-openai chromadb langchain
$ poetry install
```

---

将上述逻辑添加到新文件`chain.py`中。

---

添加到`server.py` -

```
from app.chain import chain as nomic_chain
add_routes(app, nomic_chain, path="/nomic-rag")
```

运行 -
```
$ poetry run langchain serve
```

```python
```
”