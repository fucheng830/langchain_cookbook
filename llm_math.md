“# 数学链

本笔记本展示了使用LLMs和Python REPLs解决复杂文字数学问题的方法。


```python
from langchain.chains import LLMMathChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
llm_math = LLMMathChain.from_llm(llm, verbose=True)

llm_math.invoke("13的0.3432次方是多少？")
```

    
    
    [1m> 进入新的LLMMathChain链...[0m
    13的0.3432次方是多少？[32;1m[1;3m
    ```text
    13 ** 0.3432
    ```
    ...numexpr.evaluate("13 ** 0.3432")...
    [0m
    答案： [33;1m[1;3m2.4116004626599237[0m
    [1m> 链结束。[0m





    '答案： 2.4116004626599237'




```python

```

”