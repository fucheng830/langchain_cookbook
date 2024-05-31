# 分析一篇长文档

AnalyzeDocumentChain 接受一篇文档，将其拆分，然后通过 CombineDocumentsChain 处理。

```python
with open("../docs/docs/modules/state_of_the_union.txt") as f:
    state_of_the_union = f.read()
```

```python
from langchain.chains import AnalyzeDocumentChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
```

```python
from langchain.chains.question_answering import load_qa_chain

qa_chain = load_qa_chain(llm, chain_type="map_reduce")
```

```python
qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
```

```python
qa_document_chain.run(
    input_document=state_of_the_union,
    question="总统在谈到布雷耶大法官时说了什么？",
)
```

    '总统说：“今晚，我想向一个致力于为国家服务的人表示敬意：退休的美国最高法院大法官斯蒂芬·布雷耶——一名陆军退伍军人、宪法学者。布雷耶大法官，感谢您的服务。”'