“# 基于Qianfan和BES的RAG

本笔记本是使用百度Qianfan平台结合百度ElasticSearch实现检索增强生成（RAG）的实现，原始数据存储在BOS上。

## 百度Qianfan
百度AI云Qianfan平台是为企业开发者提供的一站式大型模型开发与服务运营平台。Qianfan不仅提供包括文心一言（ERNIE-Bot）模型和第三方开源模型，还提供各种AI开发工具和完整的开发环境，方便客户轻松使用和开发大型模型应用。

## 百度ElasticSearch
[百度云VectorSearch](https://cloud.baidu.com/doc/BES/index.html?from=productToDoc)是一个完全托管的企业级分布式搜索和分析服务，与开源100%兼容。百度云VectorSearch为结构化和非结构化数据提供低成本、高性能和可靠的检索和分析平台级产品服务。作为一个向量数据库，它支持多种索引类型和相似度距离方法。

## 安装与设置



```python
#!pip install qianfan
#!pip install bce-python-sdk
#!pip install elasticsearch == 7.11.0
#!pip install sentence-transformers
```

## 导入


```python
import sentence_transformers
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
from langchain.chains.retrieval_qa import RetrievalQA
from langchain_community.document_loaders.baiducloud_bos_directory import (
    BaiduBOSDirectoryLoader,
)
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.baidu_qianfan_endpoint import QianfanLLMEndpoint
from langchain_community.vectorstores import BESVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

## 文档加载


```python
bos_host = "你的BOS端点"
access_key_id = "你的BOS访问AK"
secret_access_key = "你的BOS访问SK"

# 创建BceClientConfiguration
config = BceClientConfiguration(
    credentials=BceCredentials(access_key_id, secret_access_key), endpoint=bos_host
)

loader = BaiduBOSDirectoryLoader(conf=config, bucket="llm-test", prefix="llm/")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)
```

## 嵌入和向量存储


```python
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
embeddings.client = sentence_transformers.SentenceTransformer(embeddings.model_name)

db = BESVectorStore.from_documents(
    documents=split_docs,
    embedding=embeddings,
    bes_url="你的BES URL",
    index_name="test-index",
    vector_query_field="vector",
)

db.client.indices.refresh(index="test-index")
retriever = db.as_retriever()
```

## QA检索器


```python
llm = QianfanLLMEndpoint(
    model="ERNIE-Bot",
    qianfan_ak="你的Qianfan AK",
    qianfan_sk="你的Qianfan SK",
    streaming=True,
)
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="refine", retriever=retriever, return_source_documents=True
)

query = "张量是什么?"
print(qa.run(query))
```

> 张量（Tensor）是一个数学概念，用于表示多维数据。它是一个可以表示多个数值的数组，可以是标量、向量、矩阵等。在深度学习和人工智能领域中，张量常用于表示神经网络的输入、输出和权重等。

”