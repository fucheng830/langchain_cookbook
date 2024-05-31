以下是您提供的文本的中文翻译：

```
# 在表格数据库中合并语义相似性

在本教程中，我们将介绍如何在一个SQL查询中同时运行基于语义的搜索和传统的表格查询，同时使用RAG。

### 总体工作流程

1. 为特定列生成嵌入向量
2. 将嵌入向量存储在新列中（如果列的基数较低，最好使用包含唯一值及其嵌入向量的另一个表）
3. 使用标准SQL查询结合PGVector扩展进行查询，允许使用L2距离（`<->`）、余弦距离（`<=>`或使用`1 - <=>`表示余弦相似性）和内积（`<#>`）
4. 运行标准SQL查询

### 要求

我们需要一个启用了PGVector扩展的PostgreSQL数据库。本示例中，我们将使用一个名为`Chinook`的数据库，该数据库连接到本地PostgreSQL服务器。

```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or getpass.getpass(
    "OpenAI API Key:"
)
```

```python
from langchain.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI

CONNECTION_STRING = "postgresql+psycopg2://postgres:test@localhost:5432/vectordb"  # 替换为您的连接字符串
db = SQLDatabase.from_uri(CONNECTION_STRING)
```

### 歌曲标题的嵌入

在本例中，我们将基于语义意义运行查询，以获取歌曲标题。首先，我们需要在表中添加一个新列来存储嵌入向量：

```python
# db.run('ALTER TABLE "Track" ADD COLUMN "embeddings" vector;')
```

接下来，为每个*轨道标题*生成嵌入向量，并将其插入到我们的"Track"表中：

```python
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()
```

```python
tracks = db.run('SELECT "Name" FROM "Track"')
song_titles = [s[0] for s in eval(tracks)]
title_embeddings = embeddings_model.embed_documents(song_titles)
len(title_embeddings)
```

```
3503
```

现在，让我们将嵌入向量插入到新列中：

```python
from tqdm import tqdm

for i in tqdm(range(len(title_embeddings))):
    title = song_titles[i].replace("'", "''")
    embedding = title_embeddings[i]
    sql_command = (
        f'UPDATE "Track" SET "embeddings" = ARRAY{embedding} WHERE "Name" ='
        + f"'{title}'"
    )
    db.run(sql_command)
```

我们可以通过以下查询测试语义搜索：

```python
embeded_title = embeddings_model.embed_query("hope about the future")
query = (
    'SELECT "Track"."Name" FROM "Track" WHERE "Track"."embeddings" IS NOT NULL ORDER BY "embeddings" <-> '
    + f"'{embeded_title}' LIMIT 5"
)
db.run(query)
```

```
['Tomorrow\'s Dream', 'Remember Tomorrow', 'Remember Tomorrow', 'The Best Is Yet To Come', 'Thinking \'Bout Tomorrow']
```

### 创建SQL链

让我们开始定义有用的函数来从数据库获取信息并运行查询：

```python
def get_schema(_):
    return db.get_table_info()


def run_query(query):
    return db.run(query)
```

现在让我们构建我们将使用的**提示**：

```python
from langchain_core.prompts import ChatPromptTemplate

template = """你是一个Postgres专家。给定一个输入问题，首先创建一个语法正确的Postgres查询，然后查看查询结果，以回答输入问题。
除非用户在问题中