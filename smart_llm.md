# 使用 SmartLLMChain

SmartLLMChain 是一种自我批评链，可以帮助您回答一些特别复杂的问题。它的运作方式不是进行一次 LLM 处理，而是执行这三个步骤：
1. 构想：将用户提示连续传递给 LLM n 次，以获得 n 个输出提案（称为“想法”），其中 n 是一个您可以设置的参数。
2. 批评：LLM 对所有想法进行批评，以找出可能的缺陷，并选择最佳的一个。
3. 解决问题：LLM 尝试改进最佳想法（如批评步骤中选择的），并输出它。这是最终的输出。

SmartLLMChains 是基于 SmartGPT 工作流程的，该工作流程在 https://youtu.be/wVzuvf9D9BU 中提出。

请注意，SmartLLMChains
- 使用更多的 LLM 处理（即 n+2 而不是仅 1）
- 只有在底层 LLM 具有反思能力时才有效，较小模型通常没有这种能力
- 只与返回确切 1 个输出的底层模型工作，而不是多个

此笔记本演示了如何使用 SmartLLMChain。

##### 所有步骤使用相同 LLM


```python
import os

os.environ["OPENAI_API_KEY"] = "..."
```


```python
from langchain.prompts import PromptTemplate
from langchain_experimental.smart_llm import SmartLLMChain
from langchain_openai import ChatOpenAI
```

作为示例问题，我们将使用“我有 12 升的水壶和 6 升的水壶。我想测量 6 升的水。我该怎么做？”这是从原始 SmartGPT 视频中获得的示例（https://youtu.be/wVzuvf9D9BU?t=384）。尽管这个问题看起来非常简单，但即使是 GPT4 也很难回答这类涉及数字和物理推理的问题。

正如我们将看到的，三个初始想法都是完全错误的 - 尽管我们使用了 GPT4！只有通过自我反思，我们才能得到正确的答案。


```python
hard_question = "我有 a 12 升的水壶和 a 6 升的水壶。我想测量 6 升的水。我该怎么做？"
```

所以，我们首先创建一个 LLM 和提示模板


```python
prompt = PromptTemplate.from_template(hard_question)
llm = ChatOpenAI(temperature=0, model_name="gpt-4")
```

现在我们可以创建一个 SmartLLMChain


```python
chain = SmartLLMChain(llm=llm, prompt=prompt, n_ideas=3, verbose=True)
```

现在我们可以使用 SmartLLM 作为 LLM 的替代品。例如：


```python
chain.invoke({})
```

    
    
    [1m> 进入新的 SmartLLMChain 链...[0m
    格式化后的提示：
    [32;1m[1;3m我有 a 12 升的水壶和 a 6 升的水壶。我想测量 6 升的水。我该怎么做？[0m
    想法 1：
    [36;1m[1;3m1. 完全填满 6 升的水壶。
    2. 将水从 6 升的水壶倒入 12 升的水壶中。
    3. 再次填满 6 升的水壶。
    4. 小心地将水从 6 升的水壶倒入 12 升的水壶中，直到 12 升的水壶满。
    5. 6 升的水将留在 6 升的水壶中。[0m
    想法 2：
    [36;1m[1;3m1. 完全填满 6 升的水壶。
    2. 将水从 6 升的水壶倒入 12 升的水壶中。
    3. 再次填满 6 升的水壶。
    4. 小心地将水从 6 升的水壶倒入 12 升的水壶中，直到 12 升的水壶满。
    5. 由于 12 升的水壶现在已满，6 升的水将留在 6 升的水壶中。