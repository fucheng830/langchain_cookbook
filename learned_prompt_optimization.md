“# 通过强化学习学习提示变量注入

LLM 提示可以通过将特定术语注入模板句子中得到增强。选择正确的术语对于获得高质量的响应至关重要。本笔记本通过使用 VowpalWabbit 进行强化学习，介绍了通过术语注入实现自动提示工程的方法。

rl_chain（强化学习链）提供了一种方法，可以自动确定最佳术语注入，而无需对底层基础模型进行微调。

为了说明，考虑一个餐食配送服务的场景。我们使用 LangChain 向客户（如 Tom）询问他们的饮食偏好，并从我们广泛的菜单中推荐合适的餐食。rl_chain 根据用户偏好选择餐食，将其注入提示模板，并将提示转发给 LLM。LLM 的响应，即个性化推荐，然后返回给用户。

下面概述的示例是一个玩具示例，以展示该概念的适用性。在最后提供了高级选项和解释。

```python
# 安装必要的包
# ! pip install langchain langchain-experimental matplotlib vowpal_wabbit_next sentence-transformers pandas
```

```python
# 定义四种餐食，有些是素食，有些不是

meals = [
    "牛肉玉米卷配菲达奶酪。墨西哥-希腊融合",
    "鸡肉比萨饼配红酱。意大利-墨西哥融合",
    "蔬菜红薯玉米饼配纯素奶酪",
    "一锅烤意大利面配辣椒和洋葱",
]
```

```python
# 选择并配置您选择的 LLM

from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")
```

##### 使用提供的默认值初始化 RL 链

需要定义用于查询 LLM 的提示模板。它可以是任何内容，但这里使用了 `{meal}`，并将被上述餐食之一替换，RL 链将尝试选择并注入最佳餐食。

```python
from langchain.prompts import PromptTemplate

# 这里我使用了变量 meal，它将被上述餐食之一替换
# 还有一些变量，如 user、preference 和 text_to_personalize，我将在链运行时提供

PROMPT_TEMPLATE = """这里是对一餐的描述："{meal}"。

将餐食嵌入到给定的文本中："{text_to_personalize}"。

在前面加上一个包含用户名 "{user}" 和他们的偏好 "{preference}" 的个性化消息。

让它听起来不错。
"""

PROMPT = PromptTemplate(
    input_variables=["meal", "text_to_personalize", "user", "preference"],
    template=PROMPT_TEMPLATE,
)
```

接下来，初始化 RL 链的 PickBest 链。我们必须提供选择的 llm 和定义的提示。顾名思义，链的目标是根据某些标准从提供的餐食中选择最佳餐食。

```python
import langchain_experimental.rl_chain as rl_chain

chain = rl_chain.PickBest.from_llm(llm=llm, prompt=PROMPT)
```

一旦链设置好，我将使用我希望从中选择的餐食和一些基于此链将选择餐食的上下文来调用它。

```python
response = chain.run(
    meal=rl_chain.ToSelectFrom(meals),
    user=rl_chain.BasedOn("Tom"),
    preference=rl_chain.BasedOn(["素食", "常规乳制品可以"]),
    text_to_personalize="这是本周的特色菜，我们的主厨相信您会喜欢它！",
)
```

```python
print(response["response"])
```

    Hey Tom! 本周我们为您准备了一道特别的菜肴 - 我们的主厨制作了一道美味的一锅烤意大利面配辣椒和洋葱，非常适合素食者，且可以接受常规乳制品！我们知道您会喜欢它的！

## 链在做什么

以下是 RL 链操作的逐步分解：

1. 接受餐食列表。
2. 考虑用户及其饮食偏好。
3. 根据此上下文，选择适当的餐食。
4. 自动评估餐食选择的适当性。
5. 将选定的餐食注入提示并提交给 LLM。
6. 将 LLM 的响应返回给用户。

从技术上讲，链通过使用上下文强盗强化学习模型实现这一点，特别是利用了 [VowpalWabbit](https://github.com/VowpalWabbit/vowpal_wabbit) ML 库。

最初，由于 RL 模型未经过训练，它可能会选择不一定符合用户偏好的随机选项。然而，随着它更多地接触用户的