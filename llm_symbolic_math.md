“# LLM 符号数学
本笔记本展示了使用LLMs和Python求解代数方程。其背后使用了[SymPy](https://www.sympy.org/en/index.html)。

```python
from langchain_experimental.llm_symbolic_math.base import LLMSymbolicMathChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
llm_symbolic_math = LLMSymbolicMathChain.from_llm(llm)
```

## 积分和导数

```python
llm_symbolic_math.invoke("求sin(x)*exp(x)关于x的导数是多少？")
```

    答案：exp(x)*sin(x) + exp(x)*cos(x)

```python
llm_symbolic_math.invoke("求exp(x)*sin(x) + exp(x)*cos(x)关于x的积分是多少？")
```

    答案：exp(x)*sin(x)

## 解线性和微分方程

```python
llm_symbolic_math.invoke('求解微分方程 y" - y = e^t')
```

    答案：Eq(y(t), C2*exp(-t) + (C1 + t/2)*exp(t))

```python
llm_symbolic_math.invoke("求解方程 y^3 + 1/3y 的解是什么？")
```

    答案：{0, -sqrt(3)*I/3, sqrt(3)*I/3}

```python
llm_symbolic_math.invoke("x = y + 5, y = z - 3, z = x * y. 求解x, y, z")
```

    答案：(3 - sqrt(7), -sqrt(7) - 2, 1 - sqrt(7)), (sqrt(7) + 3, -2 + sqrt(7), 1 + sqrt(7))
```