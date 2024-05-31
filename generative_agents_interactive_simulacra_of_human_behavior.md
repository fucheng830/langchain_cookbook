“
# 基于LangChain的生成代理

此笔记本实现了基于论文《生成代理：人类行为的交互式模拟》的生成代理。

在此，我们利用了基于LangChain检索器的加权时间记忆对象。

```python
# 使用termcolor来轻松地给输出着色。
!pip install termcolor > /dev/null
```

```python
import logging

logging.basicConfig(level=logging.ERROR)
```

```python
from datetime import datetime, timedelta
from typing import List

from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from termcolor import colored
```

```python
USER_NAME = "Person A"  # 您想要在采访代理时使用的名字。
LLM = ChatOpenAI(max_tokens=1500)  # 可以任意选择LLM。
```

### 生成代理记忆组件

此教程突出了生成代理的记忆及其对行为的影响。记忆与标准的LangChain聊天记忆在两个方面有所不同：

1. **记忆形成**

   生成代理具有扩展的记忆，存储在单个流中：
      1. 观察 - 从对话或与虚拟世界的交互中获得的关于自我或其他人的观察
      2. 反思 - 重新浮现并总结的核心记忆


2. **记忆回忆**

   根据重要性、最近性和显著性，使用加权求和来检索记忆。

您可以在[参考文档]("https://api.python.langchain.com/en/latest/modules/experimental.html")中查看`GenerativeAgent`和`GenerativeAgentMemory`的定义，重点关注`add_memory`和`summarize_related_memories`方法。

## 记忆生命周期

总结了上述关键方法：`add_memory`和`summarize_related_memories`。

当代理做出观察时，它存储记忆：
    
1. 语言模型对记忆的重要性进行评分（1表示平凡，10表示深刻）
2. 观察和重要性存储在由TimeWeightedVectorStoreRetriever管理的文档中，具有`last_accessed_time`。

当代理响应观察时：

1. 生成查询（s），检索器根据显著性、最近性和重要性检索文档。
2. 总结检索到的信息
3. 更新使用文档的`last_accessed_time`。

## 创建生成角色

现在我们已经了解了定义，我们将创建两个角色“Tommie”和“Eve”。

```python
import math

import faiss


def relevance_score_fn(score: float) -> float:
    """返回一个相似度分数[0, 1]."""
    # 这将取决于几个因素：
    # - 向量存储使用的距离/相似度度量
    # - 嵌入的规模（OpenAI的是单位范数。许多其他都不是！）
    # 此函数将归一化嵌入的欧几里得范数
    # （0是最相似，sqrt(2)是最不相似）
    # 转换为相似度函数（0到1）
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """为代理创建一个新的向量存储检索器。”
    # 定义您的嵌入模型
    embeddings_model = OpenAIEmbeddings()
    # 初始化向量存储为空
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )
```

```python
tommies_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_