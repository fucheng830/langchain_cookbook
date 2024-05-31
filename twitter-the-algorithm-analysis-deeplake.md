"
# 使用 LangChain、GPT4 和 Activeloop 的 Deep Lake 分析 Twitter 算法源代码

在本教程中，我们将使用 Langchain + Activeloop 的 Deep Lake 和 GPT4 来分析 Twitter 算法的源代码。

首先，我们需要安装所需的库：

```python
!pip install langchain deeplake openai tiktoken
```

然后，定义 OpenAI 嵌入、Deep Lake 多模态向量存储 API 并认证。Deep Lake 的完整文档请参阅 [文档](https://docs.activeloop.ai/) 和 [API 参考](https://docs.deeplake.ai/en/latest/)。

```python
import getpass
import os

from langchain_community.vectorstores import DeepLake
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
activeloop_token = getpass.getpass("Activeloop Token:")
os.environ["ACTIVELOOP_TOKEN"] = activeloop_token
```

```python
embeddings = OpenAIEmbeddings(disallowed_special=())
```

disallowed_special=() 是必需的，以避免 tiktoken 对于某些仓库的 `Exception: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte`。

### 1. 索引代码库（可选）

你可以直接跳过这一部分，直接使用已索引的数据集。首先，我们将克隆仓库，然后解析和分块代码库，并使用 OpenAI 索引。

```python
!git clone https://github.com/twitter/the-algorithm # 替换为你选择的任何仓库
```

加载仓库中的所有文件

```python
import os

from langchain_community.document_loaders import TextLoader

root_dir = "./the-algorithm"
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
            docs.extend(loader.load_and_split())
        except Exception:
            pass
```

然后，分块文件

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
```

执行索引。这将花费大约 4 分钟来计算嵌入并上传到 Activeloop。然后，你可以发布数据集以使其公开。

```python
username = "<USERNAME_OR_ORG>"  # 替换为你的用户名
db = DeepLake(
    dataset_path=f"hub://{username}/twitter-algorithm",
    embedding=embeddings,
)
db.add_documents(texts)
```

### 2. 在 Twitter 算法代码库上进行问答

首先加载数据集，构建检索器，然后构建对话链。

```python
db = DeepLake(
    dataset_path=f"hub://{username}/twitter-algorithm",
    read_only=True,
    embedding=embeddings,
)
```

检索器。你可以使用 Deep Lake 的过滤器来指定用户定义的函数。

```python
def filter(x):
    # 根据源代码进行过滤
    if "com.google" in x["text"].data()["value"]:
        return False

    # 根据路径（例如扩展名）进行过滤
    metadata = x["metadata"].data()["value"]
    return "scala" in metadata["source"] or "py" in metadata["source"]

# 开启下面的自定义过滤
# retriever.search_kwargs['filter'] = filter
```

从 LangChain 构建对话链。

```python
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo