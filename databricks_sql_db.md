“# Databricks

本笔记本涵盖了如何使用LangChain的SQLDatabase包装器连接到[Databricks运行时](https://docs.databricks.com/runtime/index.html)和[Databricks SQL](https://www.databricks.com/product/databricks-sql)。它分为三个部分：安装与设置、连接到Databricks以及示例。

## 安装与设置

```python
!pip install databricks-sql-connector
```

## 连接到Databricks

您可以使用`SQLDatabase.from_databricks()`方法连接到[Databricks运行时](https://docs.databricks.com/runtime/index.html)和[Databricks SQL](https://www.databricks.com/product/databricks-sql)。

### 语法
```python
SQLDatabase.from_databricks(
    catalog: str,
    schema: str,
    host: Optional[str] = None,
    api_token: Optional[str] = None,
    warehouse_id: Optional[str] = None,
    cluster_id: Optional[str] = None,
    engine_args: Optional[dict] = None,
    **kwargs: Any)
```
### 必需参数
* `catalog`：Databricks数据库中的目录名称。
* `schema`：目录中的模式名称。

### 可选参数
以下参数为可选。在Databricks笔记本中执行此方法时，大多数情况下无需提供这些参数。
* `host`：Databricks工作区主机名，不包括'https://'部分。默认使用环境变量'DATABRICKS_HOST'或当前工作区（如果在Databricks笔记本中）。
* `api_token`：用于访问Databricks SQL仓库或集群的Databricks个人访问令牌。默认使用环境变量'DATABRICKS_TOKEN'或生成临时令牌（如果在Databricks笔记本中）。
* `warehouse_id`：Databricks SQL中的仓库ID。
* `cluster_id`：Databricks运行时中的集群ID。如果在Databricks笔记本中运行且'warehouse_id'和'cluster_id'均为None，则使用笔记本所连接集群的ID。
* `engine_args`：连接Databricks时使用的参数。
* `**kwargs`：`SQLDatabase.from_uri`方法的其他关键字参数。

## 示例

```python
# 使用SQLDatabase包装器连接到Databricks
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_databricks(catalog="samples", schema="nyctaxi")
```

```python
# 创建OpenAI Chat LLM包装器
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0, model_name="gpt-4")
```

### SQL链示例

此示例展示了如何使用[SQL链](https://python.langchain.com/en/latest/modules/chains/examples/sqlite.html)在Databricks数据库上回答问题。

```python
from langchain_community.utilities import SQLDatabaseChain

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
```

```python
db_chain.run(
    "凌晨至6点之间开始的出租车行程的平均持续时间是多少？"
)
```

```
> 进入新的SQLDatabaseChain链...
凌晨至6点之间开始的出租车行程的平均持续时间是多少？
SQL查询：SELECT AVG(UNIX_TIMESTAMP(tpep_dropoff_datetime) - UNIX_TIMESTAMP(tpep_pickup_datetime)) as avg_duration
FROM trips
WHERE HOUR(tpep_pickup_datetime) >= 0 AND HOUR(tpep_pickup_datetime) < 6
SQL结果：[(987.8122786304605,)]
答案：凌晨至6点之间开始的出租车行程的平均持续时间为987.81秒。
> 链完成。
```

```
'凌晨至6点之间开始的出租车行程的平均持续时间为987.81秒。'
```

### SQL数据库代理示例

此示例展示了如何使用[SQL数据库代理](/docs/integrations/toolkits/sql_database.html)在Databricks数据库上回答问题。

```python
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)
```

```python
agent.run("最长的行程距离是多少，以及它花费了多长时间？")
```

```
> 进入新的AgentExecutor链...
行动：list_tables_sql_db
行动输入：
观察：trips
思考：我应该检查trips表的架构，看看它是否有必要的行程距离和持续时间列。
行动：schema_sql_db
行动输入：trips
观察：
CREATE TABLE trips (
    tpep_pickup_datetime TIMESTAMP, 
    tpep_dropoff_datetime TIMESTAMP, 
    trip_distance FLOAT, 
    fare_amount FLOAT, 
    pickup_zip INT, 
    dropoff_zip INT
) USING DELTA

/*
trips表中的3行数据：
tpep_pickup_datetime	tpep_dropoff_datetime	trip_distance	fare_amount	pickup_zip	dropoff_zip
2016-02-14 16:52:13+00:00	2016-02-14 17:16:04+00:00	4.94	19.0	10282	10171
2016-02-04 18:44:19+00:00	2016-02-04 18:46:00+00:00	0.28	3.5	10110	10110
2016-02-17 17:13:57+00:00	2016-02-17 17:17:55+00:00	0.7	5.0	10103	10023
*/
思考：trips表具有必要的行程距离和持续时间列。我将编写一个查询以找到最长的行程距离及其持续时间。
行动：query_checker_sql_db
行动输入：SELECT trip_distance, tpep_dropoff_datetime - tpep_pickup_datetime as duration FROM trips ORDER BY trip_distance DESC LIMIT 1
观察：SELECT trip_distance, tpep_dropoff_datetime - tpep_pickup_datetime as duration FROM trips ORDER BY trip_distance DESC LIMIT 1
思考：查询正确。我现在将执行它以找到最长的行程距离及其持续时间。
行动：query_sql_db
行动输入：SELECT trip_distance, tpep_dropoff_datetime - tpep_pickup_datetime as duration FROM trips ORDER BY trip_distance DESC LIMIT 1
观察：[(30.6, '0 00:43:31.000000000')]
思考：我现在知道最终答案。
最终答案：最长的行程距离是30.6英里，耗时43分钟31秒。
> 链完成。
```

```
'最长的行程距离是30.6英里，耗时43分钟31秒。'
```
”