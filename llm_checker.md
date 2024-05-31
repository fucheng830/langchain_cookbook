“
# 自检链
这个笔记本展示了如何使用LLMCheckerChain。


```python
from langchain.chains import LLMCheckerChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0.7)

text = "什么类型的哺乳动物下最大的蛋？"

checker_chain = LLMCheckerChain.from_llm(llm, verbose=True)

checker_chain.invoke(text)
```

    
    
    [1m> 进入新的LLMCheckerChain链...[0m
    
    
    [1m> 进入新的SequentialChain链...[0m
    
    [1m> 完成链。[0m
    
    [1m> 完成链。[0m





    '没有哺乳动物下最大的蛋。大象鸟，这是一种巨大的鸟类，下了鸟类中最大的蛋。”




```python

```


”