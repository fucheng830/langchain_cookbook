“# 使用Activeloop的DeepLake进行问答

在本教程中，我们将使用Langchain + Activeloap的Deep Lake与GPT4结合，对群聊进行语义搜索和提问。

观看实际演示请点击[这里](https://twitter.com/thisissukh_/status/1647223328363679745)

## 1. 安装所需包

```python
!python3 -m pip install --upgrade langchain 'deeplake[enterprise]' openai tiktoken
```

## 2. 添加API密钥

```python
import getpass
import os

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import DeepLake
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
activeloop_token = getpass.getpass("Activeloop Token:")
os.environ["ACTIVELOOP_TOKEN"] = activeloop_token
os.environ["ACTIVELOOP_ORG"] = getpass.getpass("Activeloop Org:")

org_id = os.environ["ACTIVELOOP_ORG"]
embeddings = OpenAIEmbeddings()

dataset_path = "hub://" + org_id + "/data"
```

## 2. 创建示例数据

您可以使用以下提示通过ChatGPT生成一个群聊对话样本：

```
生成一个群聊对话，其中三个朋友谈论他们的一天，提及真实地点和虚构名字。尽量使其幽默且详细。
```

我已经生成了这样一个聊天记录，保存在`messages.txt`中。为了简化，我们将使用这个例子。

## 3. 导入聊天嵌入

我们加载文本文件中的消息，分块并上传到ActiveLoop向量存储。

```python
with open("messages.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
pages = text_splitter.split_text(state_of_the_union)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.create_documents(pages)

print(texts)

dataset_path = "hub://" + org_id + "/data"
embeddings = OpenAIEmbeddings()
db = DeepLake.from_documents(
    texts, embeddings, dataset_path=dataset_path, overwrite=True
)
```

```
您的Deep Lake数据集已成功创建！

Dataset(path='hub://adilkhan/data', tensors=['embedding', 'id', 'metadata', 'text'])
  tensor      htype      shape     dtype  compression
  -------    -------    -------   -------  ------- 
 embedding  embedding  (3, 1536)  float32   None   
    id        text      (3, 1)      str     None   
 metadata     json      (3, 1)      str     None   
   text       text      (3, 1)      str     None   

可选：您也可以使用Deep Lake的托管张量数据库作为托管服务并在那里运行查询。为此，在创建向量存储时必须将运行时参数指定为{'tensor_db': True}。此配置使查询能够在托管张量数据库上执行，而不是在客户端执行。需要注意的是，此功能不适用于存储在本地或内存中的数据集。如果向量存储已经在托管张量数据库之外创建，可以按照规定的步骤将其转移到托管张量数据库。

```python
# with open("messages.txt") as f:
#     state_of_the_union = f.read()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# pages = text_splitter.split_text(state_of_the_union)

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# texts = text_splitter.create_documents(pages)

# print(texts)

# dataset_path = "hub://" + org + "/data"
# embeddings = OpenAIEmbeddings()
# db = DeepLake.from_documents(
#     texts, embeddings, dataset_path=dataset_path, overwrite=True, runtime={"tensor_db": True}
# )
```

## 4. 提问

现在我们可以提出一个问题并使用语义搜索得到答案：

```python
db = DeepLake(dataset_path=dataset_path, read_only=True, embedding=embeddings)

retriever = db.as_retriever()
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["k"] = 4

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=False
)

# 群组讨论的餐厅叫什么名字？
query = input("输入查询:")

# 饥饿的龙虾
ans = qa({"query": query})

print(ans)
```
”