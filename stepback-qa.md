“# 后退提示（问题回答）

一种称为“后退提示”的技术可以通过首先提出一个“后退”问题来提高对复杂问题的表现。这可以与常规问题回答应用程序结合使用，然后对原始问题和后退问题都进行检索。

阅读论文 [这里](https://arxiv.org/abs/2310.06117)

查看Cobus Greyling关于此的优秀博客文章 [这里](https://cobusgreyling.medium.com/a-new-prompt-engineering-technique-has-been-introduced-called-step-back-prompting-b00e8954cacb)

在本食谱中，我们将复制此技术。我们稍微修改了使用的提示，以更好地与聊天模型配合使用。

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
```

```python
# 少量示例
examples = [
    {
        "input": "The Police乐队的成员能合法逮捕人吗？",
        "output": "The Police乐队的成员能做什么？",
    },
    {
        "input": "Jan Sindel是在哪个国家出生的？",
        "output": "Jan Sindel的个人历史是怎样的？",
    },
]
# 现在我们将这些转换为示例消息
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
```

```python
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """您是世界知识专家。您的任务是后退一步，将问题重述为一个更通用的后退问题，这更容易回答。以下是几个示例：""",
        ),
        # 少量示例
        few_shot_prompt,
        # 新问题
        ("user", "{question}"),
    ]
)
```

```python
question_gen = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
```

```python
question = "ChatGPT在特朗普担任总统期间是否存在？"
```

```python
question_gen.invoke({"question": question})
```

```
'ChatGPT是什么时候开发的？'
```

```python
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

search = DuckDuckGoSearchAPIWrapper(max_results=4)

def retriever(query):
    return search.run(query)
```

```python
retriever(question)
```

```
'这包括关于前总统唐纳德·特朗普的内容。根据进一步的测试，ChatGPT成功地为所有最近的美国总统写了赞美诗，但当我们输入一个查询...周三，一位Twitter用户发布了截图，他要求OpenAI的聊天机器人ChatGPT为前总统唐纳德·特朗普写一首积极的诗，聊天机器人拒绝了，理由是...尽管在许多方面令人印象深刻，ChatGPT也有一些重大缺陷。...[总统的名字],"拒绝为前总统特朗普写诗，但为拜登总统写了一首...在特朗普政府期间，Altman作为总统的直言不讳的批评者获得了新的关注。正是在这种背景下，有传言称他正在考虑竞选加州州长。'
```

```python
retriever(question_gen.invoke({"question": question}))
```

```
"Will Douglas Heaven 2023年3月3日 Stephanie Arnett/MITTR | Envato 当OpenAI在2022年11月底推出ChatGPT时，这家位于旧金山的人工智能公司... ChatGPT，全称Chat Generative Pre-trained Transformer，是基于大型语言模型的聊天机器人，由OpenAI开发并于2022年11月30日发布，允许用户将对话调整到所需的长度、格式、风格、细节级别和语言。ChatGPT是一种基于OpenAI基础大型语言模型（如GPT-4及其前身）构建的人工智能（AI）聊天机器人。这个聊天机器人重新定义了...2023年6月4日 ⋅ 4分钟阅读 124 SHARES 13K 在2022年底，OpenAI向世界介绍了ChatGPT。自推出以来，ChatGPT在开发新...方面没有显示出明显放缓的迹象。"
```

```python
# response_prompt_template = """您是世界知识专家。我将向您提出一个问题。您的回答应该是全面的，并且如果以下上下文相关，不应与它们矛盾。否则，如果它们不相关，请忽略它们。

# {normal_context}
# {step_back_context}

# 原始问题：{question}
# 答案："""
# response_prompt = ChatPromptTemplate.from_template(response_prompt_template)
```

```python
from langchain import hub

response_prompt = hub.pull("langchain-ai/stepback-answer")
```

```python
chain = (
    {
        # 使用正常问题检索上下文
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # 使用后退问题检索上下文
        "step_back_context": question_gen | retriever,
        # 传递问题
        "question": lambda x: x["question"],
    }
    | response_prompt
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)
```

```python
chain.invoke({"question": question})
```

```
'不，ChatGPT在唐纳德·特朗普担任总统期间并不存在。ChatGPT于2022年11月30日发布，这是在唐纳德·特朗普总统任期之后。提供的上下文提到，在特朗普政府期间，OpenAI的首席执行官Altman作为总统的直言不讳的批评者获得了关注。这表明ChatGPT在那时并未开发或可用。'
```

## 基线

```python
response_prompt_template = """您是世界知识专家。我将向您提出一个问题。您的回答应该是全面的，并且如果以下上下文相关，不应与它们矛盾。否则，如果它们不相关，请忽略它们。

{normal_context}

原始问题：{question}
答案："""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)
```

```python
chain = (
    {
        # 使用正常问题检索上下文（仅前三个结果）
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # 传递问题
        "question": lambda x: x["question"],
    }
    | response_prompt
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)
```

```python
chain.invoke({"question": question})
```

```
'是的，ChatGPT在唐纳德·特朗普担任总统期间存在。然而，重要的是要注意，您提供的具体上下文提到ChatGPT拒绝为前总统唐纳德·特朗普写一首积极的诗。这表明尽管ChatGPT在特朗普总统任期内可用，但它可能在关于他的回应方面有限制或偏见。'
```

```python

```