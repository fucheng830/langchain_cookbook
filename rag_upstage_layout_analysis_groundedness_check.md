```
# 使用 Upstage 布局分析和扎根性检查的 RAG
此示例说明了使用 Upstage 布局分析和扎根性检查的 RAG。


```python
from typing import List

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable
from langchain_upstage import (
    ChatUpstage,
    UpstageEmbeddings,
    UpstageGroundednessCheck,
    UpstageLayoutAnalysisLoader,
)

model = ChatUpstage()

files = ["/PATH/TO/YOUR/FILE.pdf", "/PATH/TO/YOUR/FILE2.pdf"]

loader = UpstageLayoutAnalysisLoader(file_path=files, split="element")

docs = loader.load()

vectorstore = DocArrayInMemorySearch.from_documents(
    docs, embedding=UpstageEmbeddings(model="solar-embedding-1-large")
)
retriever = vectorstore.as_retriever()

template = """仅根据以下上下文回答问题：
{context}

问题： {question}
"""
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

retrieved_docs = retriever.get_relevant_documents("SOLAR 模型中有多少个参数？")

groundedness_check = UpstageGroundednessCheck()
groundedness = ""
while groundedness != "grounded":
    chain: RunnableSerializable = RunnablePassthrough() | prompt | model | output_parser

    result = chain.invoke(
        {
            "context": retrieved_docs,
            "question": "SOLAR 模型中有多少个参数？",
        }
    )

    groundedness = groundedness_check.invoke(
        {
            "context": retrieved_docs,
            "answer": result,
        }
    )
```

```