以下是您提供的Python代码段的中文翻译：

```python
# 使用OpenAI工具进行提取

从未有过的简单！OpenAI的工具调用能力让文本中提取多种不同元素变得无比容易。1106号模型及以后版本支持“并行函数调用”，这使得操作变得非常简便。

```python
from typing import List, Optional
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI
```


```python
# 确保使用支持工具的最新模型
model = ChatOpenAI(model="gpt-3.5-turbo-1106")
```


```python
# Pydantic是一种轻松定义架构的方法
class Person(BaseModel):
    """关于要提取的人员的信息。”

    name: str
    age: Optional[int] = None
```


```python
chain = create_extraction_chain_pydantic(Person, model)
```


```python
chain.invoke({"input": "jane is 2 and bob is 3"})
```


```python
# 提取的结果
[Person(name='jane', age=2), Person(name='bob', age=3)]
```


```python
# 让我们定义另一个元素
class Class(BaseModel):
    """关于要提取的班级的信息。”

    teacher: str
    students: List[str]
```


```python
chain = create_extraction_chain_pydantic([Person, Class], model)
```


```python
chain.invoke({"input": "jane is 2 and bob is 3 and they are in Mrs Sampson's class"})
```


```python
# 提取的结果
[Person(name='jane', age=2),
 Person(name='bob', age=3),
 Class(teacher='Mrs Sampson', students=['jane', 'bob'])]
```

## 系统原理

在底层，这是一个简单的链式调用：

```python
from typing import Union, List, Type, Optional
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_tool
from langchain_core.runnables import Runnable
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseLanguageModel

_EXTRACTION_TEMPLATE = """提取并保存以下段落中提到的相关实体的属性和函数参数。

如果某个属性不存在且在函数参数中不是必需的，请在输出中不包括它。"""  # noqa: E501


def create_extraction_chain_pydantic(
    pydantic_schemas: Union[List[Type[BaseModel]], Type[BaseModel]],
    llm: BaseLanguageModel,
    system_message: str = _EXTRACTION_TEMPLATE,
) -> Runnable:
    if not isinstance(pydantic_schemas, list):
        pydantic_schemas = [pydantic_schemas]
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "{input}")
    ])
    tools = [convert_pydantic_to_openai_tool(p) for p in pydantic_schemas]
    model = llm.bind(tools=tools)
    chain = prompt | model | PydanticToolsParser(tools=pydantic_schemas)
    return chain
```

```python

```

这段代码描述了如何使用OpenAI的工具调用能力来从文本中提取数据。它定义了两个Pydantic模型，一个是`Person`，表示个人；另一个是`Class`，表示班级。通过调用`create_extraction_chain_pydantic`函数，可以创建一个链