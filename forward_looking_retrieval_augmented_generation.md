“# 边生成边检索的FLARE实现

本笔记本是前瞻性主动检索增强生成（FLARE）的实现。

请参阅原始仓库[这里](https://github.com/jzbjyb/FLARE/tree/main)。

基本思想是：

- 开始回答一个问题
- 如果你开始生成模型不确定的令牌，查找相关文档
- 使用这些文档继续生成
- 重复直到完成

关于如何查找相关文档有很多很酷的细节。基本上，模型不确定的令牌会被突出显示，然后调用LLM生成一个会引导到该答案的问题。例如，如果生成的文本是`Joe Biden went to Harvard`，而模型不确定的令牌是`Harvard`，那么一个好的生成问题将是`Joe Biden在哪里上的大学`。这个生成的问题然后用于检索步骤以获取相关文档。

为了设置这个链，我们将需要三样东西：

- 一个LLM来生成答案
- 一个LLM来生成用于检索的假设问题
- 一个检索器来查找答案

我们用来生成答案的LLM需要返回logprobs，以便我们可以识别不确定的令牌。因此，我们强烈建议您使用OpenAI包装器（注意：不是ChatOpenAI包装器，因为它不返回logprobs）。

我们用来生成用于检索的假设问题的LLM可以是任何东西。在这个笔记本中，我们将使用ChatOpenAI，因为它快速且便宜。

检索器可以是任何东西。在这个笔记本中，我们将使用[SERPER](https://serper.dev/)搜索引擎，因为它便宜。

其他需要了解的重要参数：

- `max_generation_len`：在检查是否有任何不确定令牌之前，要生成的最大令牌数
- `min_prob`：任何概率低于此的生成令牌将被视为不确定

## 导入

```python
import os

os.environ["SERPER_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
```

```python
from typing import Any, List

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAI
```

## 检索器

```python
class SerperSearchRetriever(BaseRetriever):
    search: GoogleSerperAPIWrapper = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        return [Document(page_content=self.search.run(query))]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError()


retriever = SerperSearchRetriever(search=GoogleSerperAPIWrapper())
```

## FLARE链

```python
# 我们设置这个以便我们可以看到确切发生了什么
from langchain.globals import set_verbose

set_verbose(True)
```

```python
from langchain.chains import FlareChain

flare = FlareChain.from_llm(
    ChatOpenAI(temperature=0),
    retriever=retriever,
    max_generation_len=164,
    min_prob=0.3,
)
```

```python
query = "详细解释langchain框架和baby agi之间的区别"
```

```python
flare.run(query)
```

```python
llm = OpenAI()
llm.invoke(query)
```

```python
flare.run("langchain和比特币的起源故事有何相似或不同之处？")
```

```python

```

”
将以上内容翻译成中文