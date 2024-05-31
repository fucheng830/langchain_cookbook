“# 重写-检索-阅读

**重写-检索-阅读** 是在论文 [Query Rewriting for Retrieval-Augmented Large Language Models](https://arxiv.org/pdf/2305.14283.pdf) 中提出的一种方法。

> 由于原始查询并不总是最优的，特别是在现实世界中，用于检索大型语言模型（LLM）...我们首先提示一个LLM重写查询，然后进行检索增强阅读。

我们展示了如何使用LangChain表达式语言轻松实现这一方法。

## 基线

基线RAG（**检索-阅读**）可以如下实现：

```python
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
```

```python
template = """仅基于以下上下文回答用户的问题：

<context>
{context}
</context>

问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(temperature=0)

search = DuckDuckGoSearchAPIWrapper()


def retriever(query):
    return search.run(query)
```

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

```python
simple_query = "什么是langchain？"
```

```python
chain.invoke(simple_query)
```

```python
"LangChain 是一个强大且多功能的Python库，它使开发者和研究人员能够创建、实验和分析语言模型和代理。它通过提供一组用于人工通用智能的功能，简化了基于语言的应用程序的开发。它可以用来构建聊天机器人，执行文档分析和摘要，并简化与各种大型语言模型提供商的交互。LangChain的独特之处在于它能够创建一个或多个语言模型之间的逻辑链接，称为链。它是一个开源库，提供了一个通用的接口来访问基础模型，并允许提示管理和与其他组件和工具的集成。"
```

虽然这对于格式良好的查询来说还可以，但对于更复杂的查询可能会出现问题。

```python
distracted_query = "那个Sam Bankman Fried的审判太疯狂了！什么是langchain？"
```

```python
chain.invoke(distracted_query)
```

```python
'基于给定的上下文，没有关于"langchain"的信息。'
```

这是因为检索器对于这些“分心”的查询处理得不好。

```python
retriever(distracted_query)
```

```python
'商业她是Sam Bankman-Fried案中的明星证人。她的证词是爆炸性的Gary Wang，他共同创立了FTX和Alameda Research，他说Bankman-Fried指示他改变...The Verge，在审判于10月4日开始后："Sam Bankman-Fried的辩护甚至试图赢得胜利吗？"。CBS Moneywatch，从周四开始："Sam Bankman-Fried的律师努力刺破...Sam Bankman-Fried，FTX的创始人，用一个词回应："Oof"。不到一年后，31岁的Bankman-Fried先生在曼哈顿的联邦法院受审，面对刑事指控...2023年7月19日。美国法官周三驳回了Sam Bankman-Fried律师的反对意见，并允许FTX创始人欺诈案的陪审团看到他在几天前向记者发送的一条粗俗信息...Sam Bankman-Fried，曾被誉为加密货币交易的奇才，因他创立的FTX金融交易所的崩溃而受审。Bankman-Fried被指控...'
```

## 重写-检索-阅读的实现

主要部分是一个重写器，用于重写搜索查询。

```python
template = """为回答给定问题，提供一个更好的网络搜索引擎查询，以'**'结束。问题：{x} 答案："""
rewrite_prompt = ChatPromptTemplate.from_template(template)
```

```python
from langchain import hub

rewrite_prompt = hub.pull("langchain-ai/rewrite")
```

```python
print(rewrite_prompt.template)
```

```python
"为回答给定问题，提供一个更好的网络搜索引擎查询，以'**'结束。问题 {x} 答案："
```

```python
# 解析器以移除`**`


def _parse(text):
    return text.strip('"').strip("**")
```

```python
rewriter = rewrite_prompt | ChatOpenAI(temperature=0) | StrOutputParser() | _parse
```

```python
rewriter.invoke({"x": distracted_query})
```

```python
'什么是Langchain的定义和目的？'
```

```python
rewrite_retrieve_read_chain = (
    {
        "context": {"x": RunnablePassthrough()} | rewriter | retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)
```

```python
rewrite_retrieve_read_chain.invoke(distracted_query)
```

```python
'基于给定的上下文，LangChain是一个开源框架，旨在简化使用大型语言模型（LLMs）创建应用程序的过程。它使LLM模型能够根据最新的在线信息生成响应，并简化了大量数据的组织，以便LLMs轻松访问。LangChain提供了一个标准接口用于链，与其他工具的集成，以及常见应用程序的端到端链。它是一个强大的库，简化了与各种LLM提供商的交互。LangChain的独特之处在于它能够创建一个或多个LLMs之间的逻辑链接，称为链。它是一个具有简化基于语言的应用程序开发功能的AI框架，并提供了一组用于人工通用智能的功能。然而，上下文中没有提供关于问题中提到的“Sam Bankman Fried审判”的任何信息。'
```

```python
```