“## Fireworks.AI + LangChain + RAG

[Fireworks AI](https://python.langchain.com/docs/integrations/llms/fireworks) 旨在提供与LangChain合作时的最佳体验，以下是Fireworks + LangChain进行RAG（Retrieval-Augmented Generation）的一个示例。

请参阅[我们的模型页面](https://fireworks.ai/models) 查看所有模型的完整列表。在本教程中，我们使用`accounts/fireworks/models/mixtral-8x7b-instruct`进行RAG。

对于RAG的目标，我们将使用Gemma技术报告，其PDF文件可从以下链接获取：[https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)

```python
%pip install --quiet pypdf chromadb tiktoken openai 
%pip uninstall -y langchain-fireworks
%pip install --editable /mnt/disks/data/langchain/libs/partners/fireworks
```

```python
import fireworks

print(fireworks)
import fireworks.client
```

```python
# 加载
import requests
from langchain_community.document_loaders import PyPDFLoader

# 从URL下载PDF并保存到临时位置
url = "https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf"
response = requests.get(url, stream=True)
file_name = "temp_file.pdf"
with open(file_name, "wb") as pdf:
    pdf.write(response.content)

loader = PyPDFLoader(file_name)
data = loader.load()

# 分割
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# 添加到向量数据库
from langchain_community.vectorstores import Chroma
from langchain_fireworks.embeddings import FireworksEmbeddings

vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=FireworksEmbeddings(),
)

retriever = vectorstore.as_retriever()
```

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# RAG提示
template = """仅基于以下上下文回答问题：
{context}

问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
from langchain_together import Together

llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.0,
    max_tokens=2000,
    top_k=1,
)

# RAG链
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)
```

```python
chain.invoke("Mixtral的架构细节是什么？")
```

输出：

```
'回答：Mixtral的架构细节如下：
- 维度（dim）：4096
- 层数（n_layers）：32
- 每个头的维度（head_dim）：128
- 隐藏维度（hidden_dim）：14336
- 头数（n_heads）：32
- KV头数（n_kv_heads）：8
- 上下文长度（context_len）：32768
- 词汇量大小（vocab_size）：32000
- 专家数量（num_experts）：8
- 顶级专家数量（top_k_experts）：2

Mixtral基于Transformer架构，并采用了与[18]中描述的相同的修改，但值得注意的是，Mixtral支持完全密集的32k令牌上下文长度，并且前馈块从一组8个不同的参数组中选择。在每一层，对于每个令牌，路由网络选择这些组（“专家”）中的两个来处理令牌并将它们的输出相加。这种方法在控制成本和延迟的同时增加了模型的参数数量，因为模型仅对每个令牌使用总参数集的一小部分。Mixtral使用32k令牌的上下文大小进行多语言数据预训练。它在多个基准测试中与Llama 2 70B和GPT-3.5相匹配或超越。特别是，Mixtral在数学、代码生成和多语言基准测试中大大优于Llama 2 70B。'

追踪：

https://smith.langchain.com/public/935fd642-06a6-4b42-98e3-6074f93115cd/r
```