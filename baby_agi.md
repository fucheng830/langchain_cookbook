# BabyAGI 用户指南

本笔记本演示了如何通过[Yohei Nakajima](https://twitter.com/yoheinakajima)的[BabyAGI](https://github.com/yoheinakajima/babyagi/tree/main)实现一个AI代理，该代理能够根据给定的目标生成并模拟执行任务。

本指南将帮助您理解创建自己的递归代理所需的组件。

尽管BabyAGI使用了特定的向量存储/模型提供者（Pinecone, OpenAI），但使用LangChain实现它的一个好处是您可以轻松地将其替换为不同的选项。在本实现中，我们使用了FAISS向量存储（因为它可以在本地运行且免费）。

## 安装和导入所需模块

```python
from typing import Optional

from langchain_experimental.autonomous_agents import BabyAGI
from langchain_openai import OpenAI, OpenAIEmbeddings
```

## 连接到向量存储

根据您使用的向量存储，此步骤可能会有所不同。

```python
from langchain.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
```

```python
# 定义您的嵌入模型
embeddings_model = OpenAIEmbeddings()
# 初始化向量存储为空
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
```

### 运行BabyAGI

现在是时候创建BabyAGI控制器并观察它尝试完成您的目标了。

```python
OBJECTIVE = "撰写今天旧金山的天气报告"
```

```python
llm = OpenAI(temperature=0)
```

```python
# LLMChains的日志记录
verbose = False
# 如果为None，将无限期继续
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
)
```

```python
baby_agi({"objective": OBJECTIVE})
```

```python
```

以上是BabyAGI用户指南的中文翻译。