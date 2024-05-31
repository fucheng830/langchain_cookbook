“
# 根据上下文长度选择LLM

不同的LLM具有不同的上下文长度。作为一个非常直接和实用的例子，OpenAI有GPT-3.5-Turbo的两个版本：一个具有4k上下文，另一个具有16k上下文。这个笔记本展示了如何根据输入路由它们。


```python
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_openai import ChatOpenAI
```


```python
short_context_model = ChatOpenAI(model="gpt-3.5-turbo")
long_context_model = ChatOpenAI(model="gpt-3.5-turbo-16k")
```


```python
def get_context_length(prompt: PromptValue):
    messages = prompt.to_messages()
    tokens = short_context_model.get_num_tokens_from_messages(messages)
    return tokens
```


```python
prompt = PromptTemplate.from_template("摘要这段文字：{context}")
```


```python
def choose_model(prompt: PromptValue):
    context_len = get_context_length(prompt)
    if context_len < 30:
        print("短期模型")
        return short_context_model
    else:
        print("长期模型")
        return long_context_model
```


```python
chain = prompt | choose_model | StrOutputParser()
```


```python
chain.invoke({"context": "一只青蛙去了池塘"})
```

    短期模型





    '这段文字提到一只青蛙访问了一个池塘。'




```python
chain.invoke(
    {"context": "一只青蛙去了池塘，坐在一条木头上，然后去了另一个池塘"}
)
```

    长期模型





    '这段文字描述了一只青蛙从一个池塘移动到另一个池塘，并在一条木头上栖息。'




```python

```


”