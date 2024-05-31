“# 检索增强生成（RAG）

本笔记本展示了一个使用[LangChain](https://www.langchain.com/)开发检索增强生成（RAG）模式的示例。它使用Azure AI文档智能作为文档加载器，可以从PDF、图像、Office和HTML文件中提取表格、段落和布局信息。输出的Markdown可以用于LangChain的Markdown标题分割器，实现文档的语义分块。然后，分块的文档被索引到Azure AI搜索向量存储中。给定用户查询，它将使用Azure AI搜索获取相关块，然后将上下文与查询一起输入以生成答案。

![语义分块-RAG.png](semantic-chunking-rag.png)


## 先决条件
- 在三个预览区域（**美国东部**、**美国西部2**、**西欧**）中的一个拥有Azure AI文档智能资源 - 如果您没有，请按照[此文档](https://learn.microsoft.com/azure/ai-services/document-intelligence/create-document-intelligence-resource?view=doc-intel-4.0.0)创建一个。
- Azure AI搜索资源 - 如果您没有，请按照[此文档](https://learn.microsoft.com/azure/search/search-create-service-portal)创建一个。
- Azure OpenAI资源和用于嵌入模型和聊天模型的部署 - 如果您没有，请按照[此文档](https://learn.microsoft.com/azure/ai-services/openai/how-to/create-resource?pivots=web-portal)创建一个。

在本演练中，我们将使用Azure OpenAI聊天模型和嵌入以及Azure AI搜索，但这里展示的所有内容都适用于任何ChatModel或LLM、嵌入和向量存储或检索器。

## 设置


```python
! pip install python-dotenv langchain langchain-community langchain-openai langchainhub openai tiktoken azure-ai-documentintelligence azure-identity azure-search-documents==11.4.0b8
```


```python
"""
此代码使用`dotenv`库加载环境变量，并设置Azure服务的必要环境变量。
环境变量从与本笔记本同目录下的`.env`文件中加载。
"""
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
```


```python
from langchain import hub
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores.azuresearch import AzureSearch
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
```

## 加载文档并将其分割成语义块


```python
# 初始化Azure AI文档智能以加载文档。您可以指定file_path或url_path来加载文档。
loader = AzureAIDocumentIntelligenceLoader(
    file_path="<您的文件路径>",
    api_key=doc_intelligence_key,
    api_endpoint=doc_intelligence_endpoint,
    api_model="预建布局",
)
docs = loader.load()

# 基于Markdown标题将文档分割成块。
要分割的标题 = [
    ("#", "一级标题"),
    ("##", "二级标题"),
    ("###", "三级标题"),
]
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=要分割的标题)

docs_string = docs[0].page_content
splits = text_splitter.split_text(docs_string)

print("分割数量: " + str(len(splits)))
```

## 嵌入并索引块


```python
# 嵌入分割后的文档并插入到Azure搜索向量存储中

aoai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="<Azure OpenAI嵌入模型>",
    openai_api_version="<Azure OpenAI API版本>",  # 例如，"2023-07-01-预览"
)

向量存储地址: str = os.getenv("AZURE_SEARCH_ENDPOINT")
向量存储密码: str = os.getenv("AZURE_SEARCH_ADMIN_KEY")

索引名称: str = "<您的索引名称>"
向量存储: AzureSearch = AzureSearch(
    azure_搜索_endpoint=向量存储地址,
    azure_搜索_key=向量存储密码,
    index_name=索引名称,
    embedding_function=aoai_embeddings.embed_query,
)

向量存储.add_documents(documents=splits)
```

## 基于问题检索相关块


```python
# 基于问题检索相关块

retriever = 向量存储.as_retriever(search_type="相似性", search_kwargs={"k": 3})

retrieved_docs = retriever.invoke("<您的问题>")

print(retrieved_docs[0].page_content)

# 使用已检入LangChain提示库的RAG提示（https://smith.langchain.com/hub/rlm/rag-prompt?organizationId=989ad331-949f-4bac-9694-660074a208a7）
prompt = hub.pull("rlm/rag-prompt")
llm = AzureChatOpenAI(
    openai_api_version="<Azure OpenAI API版本>",  # 例如，"2023-07-01-预览"
    azure_deployment="<您的聊天模型部署名称>",
    temperature=0,
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

## 文档问答


```python
# 询问关于文档的问题

rag_chain.invoke("<您的问题>")
```

## 带参考的文档问答


```python
# 返回检索到的文档或文档的特定源元数据

from operator import itemgetter

from langchain.schema.runnable import RunnableMap

rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain_with_source = RunnableMap(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "documents": lambda input: [doc.metadata for doc in input["documents"]],
    "answer": rag_chain_from_docs,
}

rag_chain_with_source.invoke("<您的问题>")
```

”