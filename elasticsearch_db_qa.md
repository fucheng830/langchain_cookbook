“# Elasticsearch

[![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/use_cases/qa_structured/integrations/elasticsearch.ipynb)

我们可以使用大型语言模型（LLMs）以自然语言与Elasticsearch分析数据库进行交互。

此链通过Elasticsearch DSL API（过滤器和聚合）构建搜索查询。

Elasticsearch客户端必须具有索引列表、映射描述和搜索查询的权限。

有关如何在本地运行Elasticsearch的说明，请参见[此处](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html)。

```python
! pip install langchain langchain-experimental openai elasticsearch

# 设置环境变量OPENAI_API_KEY或从.env文件加载
# import dotenv

# dotenv.load_dotenv()
```

```python
from elasticsearch import Elasticsearch
from langchain.chains.elasticsearch_database import ElasticsearchDatabaseChain
from langchain_openai import ChatOpenAI
```

```python
# 初始化Elasticsearch Python客户端。
# 参见 https://elasticsearch-py.readthedocs.io/en/v8.8.2/api.html#elasticsearch.Elasticsearch
ELASTIC_SEARCH_SERVER = "https://elastic:pass@localhost:9200"
db = Elasticsearch(ELASTIC_SEARCH_SERVER)
```

取消注释下一个单元格以初始化您的数据库。

```python
# customers = [
#     {"firstname": "Jennifer", "lastname": "Walters"},
#     {"firstname": "Monica","lastname":"Rambeau"},
#     {"firstname": "Carol","lastname":"Danvers"},
#     {"firstname": "Wanda","lastname":"Maximoff"},
#     {"firstname": "Jennifer","lastname":"Takeda"},
# ]
# for i, customer in enumerate(customers):
#     db.create(index="customers", document=customer, id=i)
```

```python
llm = ChatOpenAI(model="gpt-4", temperature=0)
chain = ElasticsearchDatabaseChain.from_llm(llm=llm, database=db, verbose=True)
```

```python
question = "所有客户的姓名是什么？"
chain.run(question)
```

我们可以自定义提示。

```python
from langchain.prompts.prompt import PromptTemplate

PROMPT_TEMPLATE = """给定一个问题，创建一个语法正确的Elasticsearch查询来运行。除非用户在问题中指定他们希望获得的具体示例数量，否则始终将查询限制为最多{top_k}个结果。您可以按相关列对结果进行排序，以返回数据库中最有趣的示例。

除非被告知，否则不要查询特定索引中的所有列，只询问与问题相关的少数列。

注意只使用您可以在映射描述中看到的列名。小心不要查询不存在的列。还要注意哪个列在哪个索引中。将查询作为有效的json返回。

使用以下格式：

问题：问题在这里
ES查询：格式化为json的Elasticsearch查询
"""

PROMPT = PromptTemplate.from_template(
    PROMPT_TEMPLATE,
)
chain = ElasticsearchDatabaseChain.from_llm(llm=llm, database=db, query_prompt=PROMPT)
```

”