“
# 使用LangChain、GPT和Activeloop的Deep Lake处理代码库
在本教程中，我们将使用LangChain、GPT和Activeloop的Deep Lake来分析LangChain本身的代码库。

## 设计

1. 准备数据：
   1. 使用`langchain_community.document_loaders.TextLoader`上传所有Python项目文件。我们将这些文件称为**文档**。
   2. 使用`langchain_text_splitters.CharacterTextSplitter`将所有文档分割成块。
   3. 使用`langchain.embeddings.openai.OpenAIEmbeddings`和`langchain_community.vectorstores.DeepLake`将块嵌入并上传到Deep Lake。
2. 问答：
   1. 从`langchain.chat_models.ChatOpenAI`和`langchain.chains.ConversationalRetrievalChain`构建一个链。
   2. 准备问题。
   3. 运行链以获取答案。


## 实现

### 集成准备

我们需要为外部服务设置密钥并安装必要的Python库。


```python
#!python3 -m pip install --upgrade langchain deeplake openai
```

为OpenAI嵌入、Deep Lake多模态向量存储API和认证设置。

关于Deep Lake的完整文档，请参阅https://docs.activeloop.ai/ 和API参考 https://docs.deeplake.ai/en/latest/


```python
import os
from getpass import getpass

os.environ["OPENAI_API_KEY"] = getpass()
# 请手动输入OpenAI密钥
```

如果要在Deep Lake上创建自己的数据集并发布，请从平台[app.activeloop.ai](https://app.activeloop.ai)获取API密钥。


```python
activeloop_token = getpass("Activeloop Token:")
os.environ["ACTIVELOOP_TOKEN"] = activeloop_token
```

### 准备数据

加载所有仓库文件。这里我们假设这个笔记本是从LangChain的fork中下载的，我们使用`langchain`仓库的Python文件。

如果你想使用不同仓库的文件，请将`root_dir`更改为你的仓库的根目录。


```python
!ls "../../../../../../libs"
```

    CITATION.cff  MIGRATE.md  README.md  libs	  poetry.toml
    LICENSE       Makefile	  docs	     poetry.lock  pyproject.toml



```python
from langchain_community.document_loaders import TextLoader

root_dir = "../../../../../../libs"

docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith(".py") and "*venv/" not in dirpath:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception:
                pass
print(f"{len(docs)}")
```

    2554


然后，将文件分割成块。


```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
print(f"{len(texts)}")
```

    Created a chunk of size 1010, which is longer than the specified 1000
    Created a chunk of size 3466, which is longer than the specified 1000
    Created a chunk of size 1375, which is longer than the specified 1000
    Created a chunk of size 1928, which is longer than the specified 1000
    Created a chunk of size 1075, which is longer than the specified 1000
