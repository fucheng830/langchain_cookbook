# 因果程序辅助语言（CPAL）链

CPAL链建立在最近的PAL之上，以阻止大型语言模型（LLM）的虚构。PAL方法的问题在于，它在处理嵌套依赖的数学问题时会虚构。CPAL的新创新在于它包含了因果结构来修复虚构问题。

原始[PR描述](https://github.com/langchain-ai/langchain/pull/6255)提供了完整的概述。

使用CPAL链，LLM将以下内容翻译为：

    "Tim买了和Cindy和Boris一样多的宠物。"
    "Cindy买了比Bill多两只的和Bob一样多的宠物。"
    "Boris买了比Ben多两只的和Beth一样多的宠物。"
    "Bill买了和奥巴马一样多的宠物。"
    "Bob买了和奥巴马一样多的宠物。"
    "Ben买了和奥巴马一样多的宠物。"
    "Beth买了和奥巴马一样多的宠物。"
    "如果奥巴马买了一只宠物，那么大家都买了多少只宠物总和？"


为：

![复杂图](/img/cpal_diagram.png)。

代码示例演示了此笔记本中的内容。

1. CPAL对抗虚构的价值：CPAL与PAL
    1.1 复杂叙事
    1.2 无法回答的数学问题
2. CPAL的三种因果图类型 ([《为什么》](https://en.wikipedia.org/wiki/The_Book_of_Why))。
    2.1 中介
    2.2 碰撞器
    2.3 混淆器


```python
from IPython.display import SVG
from langchain_experimental.cpal.base import CPALChain
from langchain_experimental.pal_chain import PALChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0, max_tokens=512)
cpal_chain = CPALChain.from_univariate_prompt(llm=llm, verbose=True)
pal_chain = PALChain.from_math_prompt(llm=llm, verbose=True)
```

## CPAL对抗虚构的价值：CPAL与PAL

与PAL一样，CPAL旨在减少LLM的虚构。

CPAL链与PAL链的不同之处在于：

CPAL添加了因果结构（或DAG）来链接实体动作（或数学表达式）。
CPAL的数学表达式建模了一个因果链，可以进行干预，而PAL的数学表达式则是投影的数学身份。

### 1.1 复杂叙事

结论：PAL会虚构，而CPAL不会虚构。

```python
question = (
    "Tim买了和Cindy和Boris一样多的宠物。"
    "Cindy买了比Bill多两只的和Bob一样多的宠物。"
    "Boris买了比Ben多两只的和Beth一样多的宠物。"
    "Bill买了和奥巴马一样多的宠物。"
    "Bob买了和奥巴马一样多的宠物。"
    "Ben买了和奥巴马一样多的宠物。"
    "Beth买了和奥巴马一样多的宠物。"
    "如果奥巴马买了一只宠物，那么大家都买了多少只宠物总和？"
)
```

```python
pal_chain.run(question)
```