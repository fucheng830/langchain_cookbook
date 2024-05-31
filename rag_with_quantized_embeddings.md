“
# 使用优化和量化嵌入器嵌入文档

在本教程中，我们将演示如何构建一个RAG管道，其中所有文档的嵌入都使用量化嵌入器完成。

我们将使用一个管道，它将：

* 创建一个文档集合。
* 使用量化嵌入器嵌入所有文档。
* 获取与我们问题相关的文档。
* 运行一个LLM来回答问题。

关于优化模型的更多信息，我们参考[optimum-intel](https://github.com/huggingface/optimum-intel.git)和[IPEX](https://github.com/intel/intel-extension-for-pytorch)。

本教程基于[Langchain RAG教程](https://towardsai.net/p/machine-learning/dense-x-retrieval-technique-in-langchain-and-llamaindex)。

```python
import uuid
from pathlib import Path

import langchain
import torch
from bs4 import BeautifulSoup as Soup
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore, LocalFileStore
from langchain_community.document_loaders.recursive_url_loader import (
    RecursiveUrlLoader,
)
from langchain_community.vectorstores import Chroma

# 对于我们的示例，我们将从网络加载文档
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCSTORE_DIR = "."
DOCSTORE_ID_KEY = "doc_id"
```

首先，我们加载这篇论文，并将其拆分为1000个字符大小的文本块。

```python
# 可以在这里添加更多的解析，因为它非常原始。
loader = RecursiveUrlLoader(
    "https://ar5iv.labs.arxiv.org/html/1706.03762",
    max_depth=2,
    extractor=lambda x: Soup(x, "html.parser").text,
)
data = loader.load()
print(f"已加载 {len(data)} 个文档")

# 拆分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
print(f"拆分为 {len(all_splits)} 个文档")
```

已加载 1 个文档
已拆分为 73 个文档

为了嵌入我们的文档，我们可以使用```QuantizedBiEncoderEmbeddings```，这是一种高效且快速的嵌入方法。

```python
from langchain_community.embeddings import QuantizedBiEncoderEmbeddings
from langchain_core.embeddings import Embeddings

model_name = "Intel/bge-small-en-v1.5-rag-int8-static"
encode_kwargs = {"normalize_embeddings": True}  # 设置为True以计算余弦相似度

model_inc = QuantizedBiEncoderEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs,
    query_instruction="Represent this sentence for searching relevant passages: ",
)
```

接下来，我们定义我们的检索器：

```python
def get_multi_vector_retriever(
    docstore_id_key: str, collection_name: str, embedding_function: Embeddings
):
    """创建组合的检索器对象。"""
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    store = InMemoryByteStore()

    return MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=docstore_id_key,
    )


retriever = get_multi_vector_retriever(DOCSTORE_ID_KEY, "multi_vec_store", model_inc)
```

接下来，我们将每个块拆分为子文档：

```python
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
id_key = "doc_id"
doc_