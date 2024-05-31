程序辅助语言模型（PAL）链

实现了程序辅助语言模型，具体见 https://arxiv.org/pdf/2211.10435.pdf。

```python
from langchain_experimental.pal_chain import PALChain
from langchain_openai import OpenAI
```

```python
llm = OpenAI(temperature=0, max_tokens=512)
```

## 数学提示

```python
pal_chain = PALChain.from_math_prompt(llm, verbose=True)
```

```python
question = "Jan有三次Marcia的宠物数量。Marcia的宠物数量比Cindy多两个。如果Cindy有四只宠物，那么这三个人总共有多少只宠物？"
```

```python
pal_chain.run(question)
```

    
    
    [1m> 进入新的PALChain链...[0m
    [32;1m[1;3mdef solution():
        """Jan有三次Marcia的宠物数量。Marcia的宠物数量比Cindy多两个。如果Cindy有四只宠物，那么这三个人总共有多少只宠物？"""
        cindy_pets = 4
        marcia_pets = cindy_pets + 2
        jan_pets = marcia_pets * 3
        total_pets = cindy_pets + marcia_pets + jan_pets
        result = total_pets
        return result[0m
    
    [1m> 完成链。[0m





    '28'



## 彩色物体

```python
pal_chain = PALChain.from_colored_object_prompt(llm, verbose=True)
```

```python
question = "在桌子上，你看到两个蓝色的小册子，两个紫色的小册子，和两个黄色的太阳镜。如果我移走桌子上所有的太阳镜，桌子上还剩下多少个紫色物品？"
```

```python
pal_chain.run(question)
```

    
    
    [1m> 进入新的PALChain链...[0m
    [32;1m[1;3m# 将物体放入列表以记录顺序
    objects = []
    objects += [('booklet', 'blue')] * 2
    objects += [('booklet', 'purple')] * 2
    objects += [('sunglasses', 'yellow')] * 2
    
    # 移走所有的太阳镜
    objects = [object for object in objects if object[0] != 'sunglasses']
    
    # 计算紫色物体的数量
    num_purple = len([object for object in objects if object[1] == 'purple'])
    answer = num_purple[0m
    
    [1m> 完成PALChain链。[0m





    '2'



## 中间步骤
你也可以使用中间步骤标志来返回生成答案的执行代码。

```python
pal_chain = PALChain.from_colored_object_prompt(
    llm, verbose=True, return_intermediate_steps=True
)
```

```python
question = "在桌子上，你看到两个蓝色的小册子，两个紫色的小册子，和两个黄色的太阳镜。如果我移走桌子上所有的太阳镜，桌子上还剩下多少个紫色物品？"
```

```python
result = pal_chain({"question": question})
```

    
    
    [1m> 进入新的PALChain链...[0m
    [32;1m[1;3m# 将物体放入列表以记录顺序
    objects = []
    objects += [('booklet', 'blue')] * 2
    objects += [('booklet', 'purple')] * 2
    objects += [('sunglasses', 'yellow')] * 2
    
    # 移走所有的太阳镜
    objects = [object for object in objects if object[0] != 'sunglasses']
    
