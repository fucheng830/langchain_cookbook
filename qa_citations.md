“# 引用提取来源

本笔记本展示了如何使用OpenAI的功能从文本中提取引用。

```python
from langchain.chains import create_citation_fuzzy_match_chain
from langchain_openai import ChatOpenAI
```

    /Users/harrisonchase/.pyenv/versions/3.9.1/envs/langchain/lib/python3.9/site-packages/deeplake/util/check_latest_version.py:32: UserWarning: 有新版本的deeplake（3.6.4）可用。建议您使用`pip install -U deeplake`更新到最新版本。
      warnings.warn(



```python
问题 = "作者在大学期间做了什么？"
上下文 = """
我叫Jason Liu，我在加拿大多伦多长大，但我出生在中国。
我上的是艺术高中，但在大学我学习了计算数学和物理。
作为实习的一部分，我在多家公司工作过，包括Stitchfix和Facebook。
我还创办了滑铁卢大学的数据科学俱乐部，并担任了两年的俱乐部主席。
"""
```


```python
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
```


```python
链 = create_citation_fuzzy_match_chain(llm)
```


```python
结果 = 链.run(问题=问题, 上下文=上下文)
```


```python
print(结果)
```

    问题='作者在大学期间做了什么？' 答案=[FactWithEvidence(事实='作者在大学学习了计算数学和物理。', 引文片段=['在大学我学习了计算数学和物理']), FactWithEvidence(事实='作者创办了滑铁卢大学的数据科学俱乐部，并担任了两年的俱乐部主席。', 引文片段=['创办了滑铁卢大学的数据科学俱乐部', '主席的俱乐部两年'])]



```python
def 高亮显示(文本, 范围):
    return (
        "..."
        + 文本[范围[0] - 20 : 范围[0]]
        + "*"
        + "\033[91m"
        + 文本[范围[0] : 范围[1]]
        + "\033[0m"
        + "*"
        + 文本[范围[1] : 范围[1] + 20]
        + "..."
    )
```


```python
for 事实 in 结果.答案:
    print("陈述:", 事实.事实)
    for 范围 in 事实.get_spans(上下文):
        print("引用:", 高亮显示(上下文, 范围))
    print()
```

    陈述: 作者在大学学习了计算数学和物理。
    引用: ...艺术高中，但*[91m在大学我学习了计算数学和物理[0m*。
    作为实习的一部分，我...
    
    陈述: 作者创办了滑铁卢大学的数据科学俱乐部，并担任了两年的俱乐部主席。
    引用: ...x和Facebook。
    我还*[91m创办了滑铁卢大学的数据科学俱乐部[0m*，并担任了主席的...
    引用: ...卢大学的数据科学俱乐部，并*[91m担任了两年的俱乐部主席[0m*。
    ...
    



```python

```

”