“# RAG融合

重新实现自[此GitHub仓库](https://github.com/Raudaschl/rag-fusion)，所有荣誉归于原作者

> RAG-Fusion，一种搜索方法，旨在弥合传统搜索范式与人类查询多维性之间的差距。受检索增强生成（RAG）能力的启发，该项目更进一步，通过使用多个查询生成和互惠排名融合来重新排序搜索结果。

## 设置

对于此示例，我们将使用Pinecone和一些假数据。要配置Pinecone，请设置以下环境变量：

- `PINECONE_API_KEY`：您的Pinecone API密钥

```python
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
```

```python
所有文档 = {
    "文档1": "气候变化与经济影响。",
    "文档2": "由于气候变化的公共卫生问题。",
    "文档3": "气候变化：社会视角。",
    "文档4": "气候变化的技术解决方案。",
    "文档5": "应对气候变化所需的政策变化。",
    "文档6": "气候变化及其对生物多样性的影响。",
    "文档7": "气候变化：科学与模型。",
    "文档8": "全球变暖：气候变化的一个子集。",
    "文档9": "气候变化如何影响日常天气。",
    "文档10": "气候变化行动主义的历史。",
}
```

```python
vectorstore = PineconeVectorStore.from_texts(
    list(所有文档.values()), OpenAIEmbeddings(), index_name="rag-fusion"
)
```

## 定义查询生成器

我们现在将定义一个用于查询生成的链

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
```

```python
from langchain import hub

prompt = hub.pull("langchain-ai/rag-fusion-query-generation")
```

```python
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是一个有帮助的助手，根据单一输入查询生成多个搜索查询。"),
#     ("user", "生成与以下内容相关的多个搜索查询：{original_query}"),
#     ("user", "输出（4个查询）：")
# ])
```

```python
generate_queries = (
    prompt | ChatOpenAI(temperature=0) | StrOutputParser() | (lambda x: x.split("\n"))
)
```

## 定义完整的链

我们现在可以将所有内容组合起来并定义完整的链。此链：
    
    1. 生成一堆查询
    2. 在检索器中查找每个查询
    3. 使用互惠排名融合将所有结果合并在一起
    
    注意，它不执行最终的生成步骤

```python
original_query = "气候变化的影响"
```

```python
vectorstore = PineconeVectorStore.from_existing_index("rag-fusion", OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
```

```python
from langchain.load import dumps, loads


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # 假设文档按相关性排序返回
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results
```

```python
chain = generate_queries | retriever.map() | reciprocal_rank_fusion
```

```python
chain.invoke({"original_query": original_query})
```




    [(Document(page_content='气候变化与经济影响。'),
      0.06558258417063283),
     (Document(page_content='气候变化：社会视角。'),
      0.06400409626216078),
     (Document(page_content='气候变化如何影响日常天气。'),
      0.04787506400409626),
     (Document(page_content='气候变化及其对生物多样性的影响。'),
      0.03306010928961749),
     (Document(page_content='由于气候变化的公共卫生问题。'),
      0.016666666666666666),
     (Document(page_content='气候变化的技术解决方案。'),
      0.016666666666666666),
     (Document(page_content='应对气候变化所需的政策变化。'),
      0.01639344262295082)]




```python
```