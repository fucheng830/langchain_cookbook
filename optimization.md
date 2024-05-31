“# 优化

本笔记本介绍了如何使用LangChain和[LangSmith](https://smith.langchain.com)优化链。

## 设置

我们将为LangSmith设置一个环境变量，并加载相关数据

```python
import os

os.environ["LANGCHAIN_PROJECT"] = "movie-qa"
```

```python
import pandas as pd
```

```python
df = pd.read_csv("data/imdb_top_1000.csv")
```

```python
df["Released_Year"] = df["Released_Year"].astype(int, errors="ignore")
```

## 创建初始检索链

我们将使用自查询检索器

```python
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
```

```python
records = df.to_dict("records")
documents = [Document(page_content=d["Overview"], metadata=d) for d in records]
```

```python
vectorstore = Chroma.from_documents(documents, embeddings)
```

```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

metadata_field_info = [
    AttributeInfo(
        name="Released_Year",
        description="电影发行的年份",
        type="int",
    ),
    AttributeInfo(
        name="Series_Title",
        description="电影的标题",
        type="str",
    ),
    AttributeInfo(
        name="Genre",
        description="电影的类型",
        type="string",
    ),
    AttributeInfo(
        name="IMDB_Rating",
        description="电影的1-10评分",
        type="float",
    ),
]
document_content_description = "电影的简短概述"
llm = ChatOpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)
```

```python
from langchain_core.runnables import RunnablePassthrough
```

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
```

```python
prompt = ChatPromptTemplate.from_template(
    """根据以下信息回答用户的问题：

信息：

{info}

问题：{question}"""
)
generator = (prompt | ChatOpenAI() | StrOutputParser()).with_config(
    run_name="generator"
)
```

```python
chain = (
    RunnablePassthrough.assign(info=(lambda x: x["question"]) | retriever) | generator
)
```

## 运行示例

通过链运行示例。可以手动进行，也可以使用示例列表或生产流量

```python
chain.invoke({"question": "2000年初期发行的恐怖电影有哪些"})
```

```python
'2000年初期发行的恐怖电影之一是"The Ring"（2002年），由Gore Verbinski执导。'
```

## 注释

现在，转到LangSmith并注释这些示例为正确或不正确

## 创建数据集

我们现在可以从那些运行中创建一个数据集。
我们将要做的是找到标记为正确的运行，然后从中抓取子链。具体来说，是查询生成子链和最终生成步骤

```python
from langsmith import Client

client = Client()
```

```python
runs = list(
    client.list_runs(
        project_name="movie-qa",
        execution_order=1,
        filter="and(eq(feedback_key, 'correctness'), eq(feedback_score, 1))",
    )
)

len(runs)
```

```python
14
```

```python
gen_runs = []
query_runs = []
for r in runs:
    gen_runs.extend(
        list(
            client.list_runs(
                project_name="movie-qa",
                filter="eq(name, 'generator')",
                trace_id=r.trace_id,
            )
        )
    )
    query_runs.extend(
        list(
            client.list_runs(
                project_name="movie-qa",
                filter="eq(name, 'query_constructor')",
                trace_id=r.trace_id,
            )
        )
    )
```

```python
runs[0].inputs
```

```python
{'question': '2000年初期发行的青春喜剧电影有哪些'}
```

```python
runs[0].outputs
```

```python
{'output': '2000年初期发行的青春喜剧电影之一是"Mean Girls"，由Lindsay Lohan、Rachel McAdams和Tina Fey主演。'}
```

```python
query_runs[0].inputs
```

```python
{'query': '2000年初期发行的青春喜剧电影有哪些'}
```

```python
query_runs[0].outputs
```

```python
{'output': {'query': '青春喜剧',
  'filter': {'operator': 'and',
   'arguments': [{'comparator': 'eq', 'attribute': 'Genre', 'value': 'comedy'},
    {'operator': 'and',
     'arguments': [{'comparator': 'gte',
       'attribute': 'Released_Year',
       'value': 2000},
      {'comparator': 'lt', 'attribute': 'Released_Year', 'value': 2010}]}]}}}
```

```python
gen_runs[0].inputs
```

```python
{'question': '2000年初期发行的青春喜剧电影有哪些', 'info': []}
```

```python
gen_runs[0].outputs
```

```python
{'output': '2000年初期发行的青春喜剧电影之一是"Mean Girls"，由Lindsay Lohan、Rachel McAdams和Tina Fey主演。'}
```

## 创建数据集

我们现在可以为查询生成和最终生成步骤创建数据集。
我们这样做是为了（1）我们可以检查数据点，（2）如果需要，我们可以编辑它们，（3）我们可以随着时间的推移添加到它们

```python
client.create_dataset("movie-query_constructor")

inputs = [r.inputs for r in query_runs]
outputs = [r.outputs for r in query_runs]

client.create_examples(
    inputs=inputs, outputs=outputs, dataset_name="movie-query_constructor"
)
```

```python
client.create_dataset("movie-generator")

inputs = [r.inputs for r in gen_runs]
outputs = [r.outputs for r in gen_runs]

client.create_examples(inputs=inputs, outputs=outputs, dataset_name="movie-generator")
```

## 使用少样本示例

我们现在可以从一个数据集拉取数据，并将其用作未来链的少样本示例

```python
examples = list(client.list_examples(dataset_name="movie-query_constructor"))
```

```python
import json


def filter_to_string(_filter):
    if "operator" in _filter:
        args = [filter_to_string(f) for f in _filter["arguments"]]
        return f"{_filter['operator']}({','.join(args)})"
    else:
        comparator = _filter["comparator"]
        attribute = json.dumps(_filter["attribute"])
        value = json.dumps(_filter["value"])
        return f"{comparator}({attribute}, {value})"
```

```python
model_examples = []

for e in examples:
    if "filter" in e.outputs["output"]:
        string_filter = filter_to_string(e.outputs["output"]["filter"])
    else:
        string_filter = "NO_FILTER"
    model_examples.append(
        (
            e.inputs["query"],
            {"query": e.outputs["output"]["query"], "filter": string_filter},
        )
    )
```

```python
retriever1 = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
    chain_kwargs={"examples": model_examples},
)
```

```python
chain1 = (
    RunnablePassthrough.assign(info=(lambda x: x["question"]) | retriever1) | generator
)
```

```python
chain1.invoke(
    {"question": "1997年至2000年之前制作的好动作电影有哪些"}
)
```

```python
'1. "Saving Private Ryan"（1998年）- 由Steven Spielberg执导，这部电影讲述了二战期间一组士兵寻找失踪伞兵的故事。\n\n2. "The Matrix"（1999年）- 由Wachowskis执导，这部科幻动作片讲述了一个发现他所生活的现实真相的计算机黑客的故事。\n\n3. "Lethal Weapon 4"（1998年）- 由Richard Donner执导，这部动作喜剧片讲述了两个不匹配的侦探调查中国移民走私团伙的故事。\n\n4. "The Fifth Element"（1997年）- 由Luc Besson执导，这部科幻动作片讲述了一个出租车司机必须保护一个神秘女人，她掌握着拯救世界的关键。\n\n5. "The Rock"（1996年）- 由Michael Bay执导，这部动作惊悚片讲述了一群叛变的军人占领了Alcatraz岛，并威胁要向旧金山发射导弹。'
```

```python
```