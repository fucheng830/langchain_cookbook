“# 摘要检查器链
本笔记本展示了一些使用不同类型文本的LLMSummarizationCheckerChain示例。与`LLMCheckerChain`相比，它有几个明显的区别，即它不对输入文本（或摘要）的格式做出任何假设。
此外，由于LLM在事实核查时喜欢产生幻觉或被上下文混淆，有时运行检查器多次是有益的。它通过将重写的“真实”结果反馈给自己，并检查“事实”的真实性来实现这一点。从下面的示例中可以看出，这在得到一个普遍真实的文本体方面非常有效。

您可以通过设置`max_checks`参数来控制检查器运行的次数。默认值为2，但如果您不希望进行双重检查，可以将其设置为1。

```python
from langchain.chains import LLMSummarizationCheckerChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
checker_chain = LLMSummarizationCheckerChain.from_llm(llm, verbose=True, max_checks=2)
text = """
您的9岁孩子可能会喜欢詹姆斯·韦伯太空望远镜（JWST）最近的这些发现：
• 在2023年，JWST发现了一批被称为“绿豌豆”的星系。它们之所以得到这个名字，是因为它们小、圆且绿色，像豌豆。
• 望远镜捕捉到了超过130亿年历史的星系图像。这意味着这些星系的光已经旅行了超过130亿年才到达我们这里。
• JWST拍摄了太阳系外行星的第一批照片。这些遥远的世界被称为“系外行星”。Exo意味着“来自外部”。
这些发现可以激发孩子对宇宙无限奇迹的想象力。"""
checker_chain.run(text)
```

...

```python
from langchain.chains import LLMSummarizationCheckerChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
checker_chain = LLMSummarizationCheckerChain.from_llm(llm, max_checks=3, verbose=True)
text = "哺乳动物可以下蛋，鸟类可以下蛋，因此鸟类是哺乳动物。"
checker_chain.run(text)
```

...

“哺乳动物不能下蛋，它们生育活体幼崽，而鸟类则下蛋。鸟类不是哺乳动物，它们属于自己独特的类别。”