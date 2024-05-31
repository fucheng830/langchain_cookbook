以下是将上述内容翻译成中文的版本：

```
# 使用OpenAI函数结构化回答

OpenAI函数允许在回答问题时结构化输出。这在回答带来源的问题时特别有用，不仅可以得到最终答案，还可以得到支持证据、引用等。

在本笔记本中，我们展示了如何使用一个基于LLM链的检索管道，该管道使用OpenAI函数作为整体检索管道的一部分。

```python
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
```

```python
loader = TextLoader("../../state_of_the_union.txt", encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
for i, text in enumerate(texts):
    text.metadata["source"] = f"{i}-pl"
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)
```

```python
from langchain.chains import create_qa_with_sources_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_openai import ChatOpenAI
```

```python
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
```

```python
qa_chain = create_qa_with_sources_chain(llm)
```

```python
doc_prompt = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
```

```python
final_qa_chain = StuffDocumentsChain(
    llm_chain=qa_chain,
    document_variable_name="context",
    document_prompt=doc_prompt,
)
```

```python
retrieval_qa = RetrievalQA(
    retriever=docsearch.as_retriever(), combine_documents_chain=final_qa_chain
)
```

```python
query = "What did the president say about russia"
```

```python
retrieval_qa.run(query)
```

```
    {
      "answer": "The President expressed strong condemnation of Russia's actions in Ukraine and announced measures to isolate Russia and provide support to Ukraine. He stated that Russia's invasion of Ukraine will have long-term consequences for Russia and emphasized the commitment to defend NATO countries. The President also mentioned taking robust action through sanctions and releasing oil reserves to mitigate gas prices. Overall, the President conveyed a message of solidarity with Ukraine and determination to protect American interests.",
      "sources": ["0-pl", "4-pl", "5-pl", "6-pl"]
    }

```

```python
qa_chain_pydantic = create_qa_with_sources_chain(llm, output_parser="pydantic")
```

```python
final_qa_chain_pydantic = StuffDocumentsChain(
    llm_chain=qa_chain_pydantic,
    document_variable_name="context",
    document_prompt=doc_prompt,
)
```

```python
retrieval_qa_pydantic = RetrievalQA(
    retriever=docsearch.as_retriever(), combine_documents_chain=final_qa_chain_pydantic
)
```

```python
retrieval_qa_pydantic.run(query)
```

```
AnswerWithSources(answer="The President expressed strong condemnation of Russia's actions in Ukraine and announced measures to isolate Russia and provide support to Ukraine. He stated that Russia's invasion of Ukraine will have long-term consequences for Russia