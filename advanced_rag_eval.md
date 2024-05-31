"
# 高级RAG评估

食谱详细介绍了在高级RAG上运行评估的过程。

这对于确定最适合您应用程序的RAG方法非常有用。

```python
! pip install -U langchain openai chromadb langchain-experimental # (需要最新版本以支持多模态)
```

```python
# 由于较新版本存在持续的bug，锁定到0.10.19
! pip install "unstructured[all-docs]==0.10.19" pillow pydantic lxml pillow matplotlib tiktoken open_clip_torch torch
```

## 数据加载

让我们查看一份[示例白皮书](https://sgp.fas.org/crs/misc/IF10244.pdf)，其中包含有关美国野火的表格、文本和图像的混合信息。

### 选项1：加载文本

```python
# 路径
path = "/Users/rlm/Desktop/cpi/"

# 加载
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(path + "cpi.pdf")
pdf_pages = loader.load()

# 分割
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits_pypdf = text_splitter.split_documents(pdf_pages)
all_splits_pypdf_texts = [d.page_content for d in all_splits_pypdf]
```

### 选项2：加载文本、表格和图像

```python
from unstructured.partition.pdf import partition_pdf

# 从PDF中提取图像、表格和文本块
raw_pdf_elements = partition_pdf(
    filename=path + "cpi.pdf",
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path,
)

# 按类型分类
tables = []
texts = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        tables.append(str(element))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        texts.append(str(element))
```

## 存储

### 选项1：嵌入、存储文本块

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

baseline = Chroma.from_texts(
    texts=all_splits_pypdf_texts,
    collection_name="baseline",
    embedding=OpenAIEmbeddings(),
)
retriever_baseline = baseline.as_retriever()
```

### 选项2：多向量检索器

#### 文本摘要

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 提示
prompt_text = """您是助手，负责总结表格和文本以供检索。 
这些摘要将被嵌入并用于检索原始文本或表格元素。 
请给出表格或文本的简洁总结，以便很好地优化检索。 表格或文本：{element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

# 文本摘要链
model = ChatOpenAI(temperature=0, model="gpt-4")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

# 应用于文本
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})

# 应用于