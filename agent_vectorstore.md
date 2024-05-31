# 将代理和向量存储结合

这个笔记本涵盖了如何将代理和向量存储结合使用。使用场景是您已经将数据导入向量存储，并希望以代理的方式与它交互。

推荐的方法是创建一个 `RetrievalQA`，然后将其作为整体代理中的工具使用。让我们看一下如何进行。您可以使用多种不同的向量数据库，并使用代理作为路由器。这里有两种不同的方法 - 您可以让代理使用向量存储作为正常工具，或者设置 `return_direct=True` 实际上只是使用代理作为路由器。

## 创建向量存储

```python
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

llm = OpenAI(temperature=0)
```

```python
from pathlib import Path

relevant_parts = []
for p in Path(".").absolute().parts:
    relevant_parts.append(p)
    if relevant_parts[-3:] == ["langchain", "docs", "modules"]:
        break
doc_path = str(Path(*relevant_parts) / "state_of_the_union.txt")
```

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader(doc_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings, collection_name="state-of-union")
```

    正在使用直接本地API运行Chroma。
    使用DuckDB内存数据库。数据将短暂存在。



```python
state_of_union = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)
```


```python
from langchain_community.document_loaders import WebBaseLoader
```


```python
loader = WebBaseLoader("https://beta.ruff.rs/docs/faq/")
```


```python
docs = loader.load()
ruff_texts = text_splitter.split_documents(docs)
ruff_db = Chroma.from_documents(ruff_texts, embeddings, collection_name="ruff")
ruff = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=ruff_db.as_retriever()
)
```

    正在使用直接本地API运行Chroma。
    使用DuckDB内存数据库。数据将短暂存在。



```python

```

## 创建代理

```python
# 导入需要的东西
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_openai import OpenAI
```

```python
tools = [
    Tool(
        name="State of Union QA System",
        func=state_of_union.run,
        description="当您需要回答关于最近一次国情咨文的提问时很有用。输入应该是完全形成的问