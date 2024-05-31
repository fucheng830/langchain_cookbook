“## LLaMA2 与 SQL 的聊天

开源、本地的大型语言模型（LLMs）非常适合需要数据隐私的应用。

SQL 就是一个很好的例子。

本食谱展示了如何使用本地运行的各种 LLaMA2 版本进行文本到 SQL 的转换。

## 包

```python
! pip install langchain replicate
```

## LLM

有几种方式可以访问 LLaMA2。

要在本地运行，我们使用 Ollama.ai。

详情请见 [此处](/docs/integrations/chat/ollama) 关于安装和设置。

此外，请参阅 [此处](/docs/guides/development/local_llms) 我们的完整本地 LLMs 指南。

要使用外部 API，这不是私有的，我们可以使用 Replicate。

```python
# 本地
from langchain_community.chat_models import ChatOllama

llama2_chat = ChatOllama(model="llama2:13b-chat")
llama2_code = ChatOllama(model="codellama:7b-instruct")

# API
from langchain_community.llms import Replicate

# REPLICATE_API_TOKEN = getpass()
# os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
replicate_id = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
llama2_chat_replicate = Replicate(
    model=replicate_id, input={"temperature": 0.01, "max_length": 500, "top_p": 1}
)
```

    Init param `input` is deprecated, please use `model_kwargs` instead.



```python
# 简单设置我们想要使用的 LLM
llm = llama2_chat
```

## 数据库

连接到 SQLite 数据库。

要创建这个特定的数据库，你可以使用代码并按照 [此处](https://github.com/facebookresearch/llama-recipes/blob/main/demo_apps/StructuredLlama.ipynb) 显示的步骤操作。

```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///nba_roster.db", sample_rows_in_table_info=0)


def get_schema(_):
    return db.get_table_info()


def run_query(query):
    return db.run(query)
```

## 查询 SQL 数据库

请按照 [此处](https://python.langchain.com/docs/expression_language/cookbook/sql_db) 的运行流程进行操作。

```python
# 提示
from langchain_core.prompts import ChatPromptTemplate

# 根据 SQL 数据库类型（如 MySQL、Microsoft SQL Server 等）更新模板
template = """根据下面的表模式，编写一个 SQL 查询来回答用户的问题：
{schema}

问题：{question}
SQL 查询："""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "给定一个输入问题，将其转换为 SQL 查询。无需前言。"),
        ("human", template),
    ]
)

# 查询链
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

sql_response.invoke({"question": "克莱·汤普森在哪个队？"})
```




    ' SELECT "Team" FROM nba_roster WHERE "NAME" = \'克莱·汤普森\';'



我们可以查看结果：

* [LangSmith 跟踪](https://smith.langchain.com/public/afa56a06-b4e2-469a-a60f-c1746e75e42b/r) LLaMA2-13 Replicate API
* [LangSmith 跟踪](https://smith.langchain.com/public/2d4ecc72-6b8f-4523-8f0b-ea95c6b54a1d/r) LLaMA2-13 本地



```python
# 回答链
template = """根据下面的表模式、问题、SQL 查询和 SQL 响应，编写一个自然语言响应：
{schema}

问题：{question}
SQL 查询：{query}
SQL 响应：{response}"""
prompt_response = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "给定一个输入问题和 SQL 响应，将其转换为自然语言答案。无需前言。",
        ),
        ("human", template),
    ]
)

full_chain = (
    RunnablePassthrough.assign(query=sql_response)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | llm
)

full_chain.invoke({"question": "有多少个不同的队伍？"})
```




    AIMessage(content=' 根据表模式和 SQL 查询，NBA 中有 30 个不同的队伍。')



我们可以查看结果：

* [LangSmith 跟踪](https://smith.langchain.com/public/10420721-746a-4806-8ecf-d6dc6399d739/r) LLaMA2-13 Replicate API
* [LangSmith 跟踪](https://smith.langchain.com/public/5265ebab-0a22-4f37-936b-3300f2dfa1c1/r) LLaMA2-13 本地

## 与 SQL 数据库聊天

接下来，我们可以添加记忆。

```python
# 提示
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = """给定一个输入问题，将其转换为 SQL 查询。无需前言。根据下面的表模式，编写一个 SQL 查询来回答用户的问题：
{schema}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

memory = ConversationBufferMemory(return_messages=True)

# 带记忆的查询链
from langchain_core.runnables import RunnableLambda

sql_chain = (
    RunnablePassthrough.assign(
        schema=get_schema,
        history=RunnableLambda(lambda x: memory.load_memory_variables(x)["history"]),
    )
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)


def save(input_output):
    output = {"output": input_output.pop("output")}
    memory.save_context(input_output, output)
    return output["output"]


sql_response_memory = RunnablePassthrough.assign(output=sql_chain) | save
sql_response_memory.invoke({"question": "克莱·汤普森在哪个队？"})
```




    ' SELECT "Team" FROM nba_roster WHERE "NAME" = \'克莱·汤普森\';'




```python
# 回答链
template = """根据下面的表模式、问题、SQL 查询和 SQL 响应，编写一个自然语言响应：
{schema}

问题：{question}
SQL 查询：{query}
SQL 响应：{response}"""
prompt_response = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "给定一个输入问题和 SQL 响应，将其转换为自然语言答案。无需前言。",
        ),
        ("human", template),
    ]
)

full_chain = (
    RunnablePassthrough.assign(query=sql_response_memory)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | llm
)

full_chain.invoke({"question": "他的薪水是多少？"})
```




    AIMessage(content=' 当然！以下是基于给定输入的自然语言响应：\n\n"克莱·汤普森的薪水是 $43,219,440."')



这里是 [跟踪](https://smith.langchain.com/public/54794d18-2337-4ce2-8b9f-3d8a2df89e51/r)。

”