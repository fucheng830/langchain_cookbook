以下是使用Anthropic API生成结构化输出的示例。

## 使用Anthropic API生成结构化输出

Anthropic API最近增加了工具使用功能。

这对于生成结构化输出非常有用。

```python
! pip install -U langchain-anthropic
```

```python
# 可选
import os
# os.environ['LANGCHAIN_TRACING_V2'] = 'true' # 启用追踪
# os.environ['LANGCHAIN_API_KEY'] = <your-api-key>
```

```python
# 如何使用工具生成结构化输出？

函数调用/工具使用只是生成负载。

负载通常是JSON字符串，可以传递给API或，在这种情况下，传递给解析器以生成结构化输出。

LangChain有`llm.with_structured_output(schema)`来非常容易地生成与`schema`匹配的结构化输出。

![Screenshot 2024-04-03 at 10.16.57 PM.png](83c97bfe-b9b2-48ef-95cf-06faeebaa048.png)

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


# 数据模型
class code(BaseModel):
    """代码输出"""

    prefix: str = Field(description="问题和解决方案的描述")
    imports: str = Field(description="代码块的导入语句")
    code: str = Field(description="代码块，不包括导入语句")


# LLM
llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    default_headers={"anthropic-beta": "tools-2024-04-04"},
)

# 结构化输出，包括原始将捕获原始输出和解析器错误
structured_llm = llm.with_structured_output(code, include_raw=True)
code_output = structured_llm.invoke(
    "编写一个Python程序，打印字符串'hello world'，并告诉我它是如何工作的。"
)
```

```python
# 初始推理阶段
code_output["raw"].content[0]
```

```python
# 工具调用
code_output["raw"].content[1]
```

```python
# JSON字符串
code_output["raw"].content[1]["input"]
```

```python
# 错误
error = code_output["parsing_error"]
error
```

```python
# 结果
parsed_result = code_output["parsed"]
```

```python
parsed_result.prefix
```

```python
# 导入
parsed_result.imports
```

```python
parsed_result.code
```

## 更复杂的示例

激励示例说明使用工具/结构化输出。

![code-gen.png](bb6c7126-7667-433f-ba50-56107b0341bd.png)

这里有一些我们想要回答代码问题的文档。

```python
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

# LCEL文档
url = "https://python.langchain.com/docs/expression_language/"
loader = RecursiveUrlLoader(
    url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

# 根据URL排序并根据URL获取文本
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)