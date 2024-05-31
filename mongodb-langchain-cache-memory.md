以下是您提供的英文内容的中文翻译：

```
# 添加语义缓存和内存到您的 RAG 应用程序中使用 MongoDB 和 LangChain

在本文中，我们将了解如何在新版的 MongoDB 中使用 MongoDBCache 和 MongoDBChatMessageHistory 功能来增强您的 RAG（Retrieval-augmented Generation）应用程序。

## 第一步：安装所需库

- **datasets**：Python 库，可以访问 Hugging Face Hub 上的数据集。
- **langchain**：Python 工具包，用于 LangChain。
- **langchain-mongodb**：Python 包，使 MongoDB 成为 LangChain 中的向量存储、语义缓存、聊天历史存储等。
- **langchain-openai**：Python 包，用于与 OpenAI 模型结合使用。
- **pymongo**：Python 工具包，用于 MongoDB。
- **pandas**：Python 库，用于数据分析、探索和操作。

```python
! pip install -qU datasets langchain langchain-mongodb langchain-openai pymongo pandas
```

## 第二步：设置 MongoDB 连接字符串

获取您的 MongoDB Atlas UI 中的连接字符串。

```python
MONGODB_URI = getpass.getpass("输入您的 MongoDB 连接字符串：")
```

## 第三步：下载数据集

我们将使用 MongoDB 的 [embedded_movies](https://huggingface.co/datasets/MongoDB/embedded_movies) 数据集。

```python
import pandas as pd
from datasets import load_dataset
```

```python
# 确保您在开发环境中有一个 HF_TOKEN：
# 访问令牌可以在 Hugging Face 平台上创建或复制（https://huggingface.co/docs/hub/en/security-tokens）

# 从 Hugging Face 加载 MongoDB 的 embedded_movies 数据集
# https://huggingface.co/datasets/MongoDB/airbnb_embeddings

data = load_dataset("MongoDB/embedded_movies")
```

## 第四步：数据分析

确保数据集的长度符合预期，丢弃 None 值等。

```python
# 查看数据集内容
df.head(1)
```

## 第五步：创建简单的 RAG 链使用 MongoDB 作为向量存储

```python
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# 初始化 MongoDB Python 客户端
client = MongoClient(MONGODB_URI, appname="devrel.content.python")

DB_NAME = "langchain_chatbot"
COLLECTION_NAME = "data"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
collection = client[DB_NAME][COLLECTION_NAME]
```

```python
# 删除集合中任何现有记录
collection.delete_many({})
```

```python
# 数据导入
records = df.to_dict("records")
collection.insert_many(records)

print("数据导入 MongoDB 完成")
```

```python
from langchain_openai import OpenAIEmbeddings

# 使用用于创建电影数据集中嵌入式的文本-嵌入-ada-002 模型
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002"
)
```

```python
# 向量存储创建
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGODB_URI,
    namespace=DB_NAME + "." + COLLECTION_NAME,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    text_key="fullplot",
)
```

```python
# 使用 MongoDB 向量存储作为 RAG 链