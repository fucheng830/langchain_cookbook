“# 使用MyScale的向量SQL检索器

>[MyScale](https://docs.myscale.com/en/)是一个集成的向量数据库。您可以通过SQL访问您的数据库，也可以通过LangChain访问。MyScale能够利用[多种数据类型和过滤函数](https://blog.myscale.com/2023/06/06/why-integrated-database-solution-can-boost-your-llm-apps/#filter-on-anything-without-constraints)。无论您是在扩展数据还是将系统扩展到更广泛的应用程序，MyScale都将加速您的LLM应用。


```python
!pip3 install clickhouse-sqlalchemy InstructorEmbedding sentence_transformers openai langchain-experimental
```


```python
import getpass
from os import environ

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql.vector_sql import VectorSQLDatabaseChain
from langchain_openai import OpenAI
from sqlalchemy import MetaData, create_engine

MYSCALE_HOST = "msc-4a9e710a.us-east-1.aws.staging.myscale.cloud"
MYSCALE_PORT = 443
MYSCALE_USER = "chatdata"
MYSCALE_PASSWORD = "myscale_rocks"
OPENAI_API_KEY = getpass.getpass("OpenAI API Key:")

engine = create_engine(
    f"clickhouse://{MYSCALE_USER}:{MYSCALE_PASSWORD}@{MYSCALE_HOST}:{MYSCALE_PORT}/default?protocol=https"
)
metadata = MetaData(bind=engine)
environ["OPENAI_API_KEY"] = OPENAI_API_KEY
```


```python
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_experimental.sql.vector_sql import VectorSQLOutputParser

output_parser = VectorSQLOutputParser.from_embeddings(
    model=HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"}
    )
)
```


```python
from langchain.callbacks import StdOutCallbackHandler
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql.prompt import MYSCALE_PROMPT
from langchain_experimental.sql.vector_sql import VectorSQLDatabaseChain
from langchain_openai import OpenAI

chain = VectorSQLDatabaseChain(
    llm_chain=LLMChain(
        llm=OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
        prompt=MYSCALE_PROMPT,
    ),
    top_k=10,
    return_direct=True,
    sql_cmd_parser=output_parser,
    database=SQLDatabase(engine, None, metadata),
)

import pandas as pd

pd.DataFrame(
    chain.run(
        "请给我10篇关于PageRank是什么的论文？",
        callbacks=[StdOutCallbackHandler()],
    )
)
```

## 作为检索器的SQL数据库


```python
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_experimental.retrievers.vector_sql_database import (
    VectorSQLDatabaseChainRetriever,
)
from langchain_experimental.sql.prompt import MYSCALE_PROMPT
from langchain_experimental.sql.vector_sql import (
    VectorSQLDatabaseChain,
    VectorSQLRetrieveAllOutputParser,
)
from langchain_openai import ChatOpenAI

output_parser_retrieve_all = VectorSQLRetrieveAllOutputParser.from_embeddings(
    output_parser.model
)

chain = VectorSQLDatabaseChain.from_llm(
    llm=OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
    prompt=MYSCALE_PROMPT,
    top_k=10,
    return_direct=True,
    db=SQLDatabase(engine, None, metadata),
    sql_cmd_parser=output_parser_retrieve_all,
    native_format=True,
)

# 您需要所有这些键来获取文档
retriever = VectorSQLDatabaseChainRetriever(
    sql_db_chain=chain, page_content_key="abstract"
)

document_with_metadata_prompt = PromptTemplate(
    input_variables=["page_content", "id", "title", "authors", "pubdate", "categories"],
    template="内容:\n\t标题: {title}\n\t摘要: {page_content}\n\t作者: {authors}\n\t出版日期: {pubdate}\n\t类别: {categories}\n来源: {id}",
)

chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(
        model_name="gpt-3.5-turbo-16k", openai_api_key=OPENAI_API_KEY, temperature=0.6
    ),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "document_prompt": document_with_metadata_prompt,
    },
    return_source_documents=True,
)
ans = chain(
    "请给我10篇关于PageRank是什么的论文？",
    callbacks=[StdOutCallbackHandler()],
)
print(ans["answer"])
```


```python

```

”