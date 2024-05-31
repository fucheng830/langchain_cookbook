"Together AI"通过推理API拥有广泛的OSS LLM。有关详细信息，请参阅[此处](https://docs.together.ai/docs/inference-models)。我们在Mixtral论文中使用`"mistralai/Mixtral-8x7B-Instruct-v0.1"`。

下载论文：
https://arxiv.org/pdf/2401.04088.pdf


```python
! pip install --quiet pypdf chromadb tiktoken openai langchain-together
```


```python
# 加载
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("~/Desktop/mixtral.pdf")
data = loader.load()

# 分割
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# 添加到向量数据库
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

"""
from langchain_together.embeddings import TogetherEmbeddings
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
"""
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)

retriever = vectorstore.as_retriever()
```


```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from lang链_core.runnables import RunnableParallel, RunnablePassthrough

# RAG提示
template = """仅根据以下上下文回答问题：
{context}

问题： {question}
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




    '\n答案：Mixtral的架构细节如下：
    - 维度（dim）：4096
    - 层数（n_layers）：32
    - 每个头的维度（head_dim）：128
    - 隐藏维度（hidden_dim）：14336
    - 头数（n_heads）：32
    - KV头数（n_kv_heads）：8
    - 上下文长度（context_len）：32768
    - 词汇表大小（vocab_size）：32000
    - 专家数（num_experts）：8
    - 顶级专家数（top_k_experts）：2
    
    Mixtral基于 transformer 架构，并使用与[18]中描述相同的修改，值得注意的是Mixtral支持32k令牌的完全密集上下文长度，并且 feedforward 块从参数组集合中选择两个组（称为“专家”）来处理令牌并将它们的输出相加。这种技术在增加模型参数的同时控制成本和延迟，因为模型仅使用每个令牌所使用参数集合的一小部分。Mixtral使用32k令牌的多元数据进行预训练。它在多个基准测试中的表现与Llama 2 70B和GPT-3.5相当或更好。特别是，Mixtral在数学、代码生成和多语言基准测试中远远超过了