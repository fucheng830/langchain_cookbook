“# 自我发现

对[自我发现论文](https://arxiv.org/pdf/2402.03620.pdf)的实现。

基于[@catid](https://github.com/catid/self-discover/tree/main?tab=readme-ov-file)的这个实现

```python
from langchain_openai import ChatOpenAI
```

```python
model = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview")
```

```python
from langchain import hub
from langchain_core.prompts import PromptTemplate
```

```python
select_prompt = hub.pull("hwchase17/self-discovery-select")
```

```python
select_prompt.pretty_print()
```

选择几个对解决给定任务至关重要的推理模块：

所有推理模块描述：
[33;1m[1;3m{reasoning_modules}[0m

任务：[33;1m[1;3m{task_description}[0m

选择几个模块对解决上述任务至关重要：

```python
adapt_prompt = hub.pull("hwchase17/self-discovery-adapt")
```

```python
adapt_prompt.pretty_print()
```

重新表述和指定每个推理模块，以便更好地帮助解决任务：

选定的模块描述：
[33;1m[1;3m{selected_modules}[0m

任务：[33;1m[1;3m{task_description}[0m

调整每个推理模块描述以更好地解决任务：

```python
structured_prompt = hub.pull("hwchase17/self-discovery-structure")
```

```python
structured_prompt.pretty_print()
```

将推理模块操作化为JSON格式的逐步推理计划：

这里有一个例子：

示例任务：

如果你遵循这些指令，你是否回到起点？始终面向前方。后退1步。向左走9步。后退2步。向前走6步。向前走4步。后退4步。向右走3步。

示例推理结构：

{
    "指令1后的位置":
    "指令2后的位置":
    "指令n后的位置":
    "最终位置是否与起始位置相同":
}

调整后的模块描述：
[33;1m[1;3m{adapted_modules}[0m

任务：[33;1m[1;3m{task_description}[0m

为求解者实施一个逐步推理结构，以便遵循并得出正确答案。

注意：在此步骤中不要实际得出结论。你的工作是生成一个计划，以便将来你可以填写它并得出此类任务的正确结论

```python
reasoning_prompt = hub.pull("hwchase17/self-discovery-reasoning")
```

```python
reasoning_prompt.pretty_print()
```

遵循JSON中的逐步推理计划，正确解决任务。根据给定的任务专门推理，填写键后面的值。不要简单地重述键。
    
推理结构：
[33;1m[1;3m{reasoning_structure}[0m

任务：[33;1m[1;3m{task_description}[0m

```python
reasoning_prompt
```

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
```

```python
select_chain = select_prompt | model | StrOutputParser()
```

```python
adapt_chain = adapt_prompt | model | StrOutputParser()
```

```python
structure_chain = structured_prompt | model | StrOutputParser()
```

```python
reasoning_chain = reasoning_prompt | model | StrOutputParser()
```

```python
overall_chain = (
    RunnablePassthrough.assign(selected_modules=select_chain)
    .assign(adapted_modules=adapt_chain)
    .assign(reasoning_structure=structure_chain)
    .assign(answer=reasoning_chain)
)
```

```python
reasoning_modules = [
    "1. 我如何设计一个实验来帮助解决这个问题？",
    "2. 为解决问题列出一系列想法，并将它们逐一应用于问题，看看是否能取得进展。",
    # "3. 我如何衡量这个问题的进展？",
    "4. 我如何简化问题以便更容易解决？",
    "5. 这个问题的关键假设是什么？",
    "6. 每个解决方案的潜在风险和缺点是什么？",
    "7. 这个问题的替代观点或视角是什么？",
    "8. 这个问题及其解决方案的长期影响是什么？",
    "9. 我如何将这个问题分解成更小、更易管理的部分？",
    "10. 批判性思维：这种风格涉及从不同角度分析问题，质疑假设，并评估可用的证据或信息。它侧重于逻辑推理、基于证据的决策制定，并识别思维中的潜在偏见或缺陷。",
    "11. 尝试创造性思维，产生创新和非常规的想法来解决问题。探索非常规解决方案，超越传统界限思考，鼓励想象力和原创性。",
    # "12. 寻求他人的意见和合作来解决问题。强调团队合作、开放沟通，并利用小组的多样化观点和专业知识来提出有效解决方案。",
    "13. 使用系统思维：将问题视为更大系统的一部分，并理解各种元素的相互联系。侧重于识别潜在原因、反馈循环和影响问题的相互依赖关系，并开发整体解决方案以解决整个系统。",
    "14. 使用风险分析：评估与不同解决方案或方法相关的潜在风险、不确定性和权衡。侧重于评估潜在后果和成功或失败的可能性，并根据风险和收益的平衡分析做出明智的决策。",
    # "15. 使用反思性思维：从问题中退一步，花时间进行内省和自我反思。检查可能影响问题解决的个人偏见、假设和心智模型，并从过去的经验中学习以改进未来的方法。",
    "16. 需要解决的核心问题或问题是什么？",
    "17. 问题的潜在原因或因素是什么？",
    "18. 是否尝试过任何潜在的解决方案或策略？如果是，结果和学到的教训是什么？",
    "19. 解决这个问题可能会遇到哪些潜在的障碍或挑战？",
    "20. 是否有任何相关数据或信息可以提供对问题的洞察？如果是，可用的数据源是什么，如何分析它们？",
    "21. 是否有任何直接受问题影响的利益相关者或个人？他们的观点和需求是什么？",
    "22. 有效解决这个问题需要哪些资源（财务、人力、技术等）？",
    "23. 如何衡量或评估解决问题的进展或成功？",
    "24. 可以使用哪些指标或度量？",
    "25. 问题是需要特定专业知识或技能集的技术或实践问题吗？还是更多的是概念或理论问题？",
    "26. 问题是否涉及物理约束，如有限资源、基础设施或空间？",
    "27. 问题是否与人类行为有关，如社会、文化或心理问题？",
    "28. 问题是否涉及决策或规划，其中需要在不确定性或竞争目标下做出选择？",
    "29. 问题是否是需要数据分析、建模或优化技术的分析问题？",
    "30. 问题是否是需要创造性解决方案和创新的设计挑战？",
    "31. 问题是否需要解决系统或结构性问题而不是个别实例？",
    "32. 问题是否是时间敏感或紧急的，需要立即关注和行动？",
    "33. 这种类型的问题规范通常会产生什么样的解决方案？",
    "34. 根据问题规范和当前最佳解决方案，猜测其他可能的解决方案。",
    "35. 让我们想象当前最佳解决方案完全错误，还有哪些其他方式可以思考问题规范？",
    "36. 根据你对这种类型问题规范的了解，修改当前最佳解决方案的最佳方式是什么？",
    "37. 忽略当前最佳解决方案，为问题创建一个全新的解决方案。",
    # "38. 让我们一步一步地思考。",
    "39. 让我们制定一个逐步计划，并使用良好的符号和解释来实施它。",
]

任务示例 = "Lisa有10个苹果。她给朋友3个苹果，然后从商店买了5个苹果。Lisa现在有多少个苹果？"

任务示例 = """这个SVG路径元素<path d="M 55.57,80.69 L 57.38,65.80 M 57.38,65.80 L 48.90,57.46 M 48.90,57.46 L
45.58,47.78 M 45.58,47.78 L 53.25,36.07 L 66.29,48.90 L 78.69,61.09 L 55.57,80.69"/>画了一个：
(A) 圆 (B) 七边形 (C) 六边形 (D) 风筝 (E) 线 (F) 八边形 (G) 五边形(H) 矩形 (I) 扇区 (J) 三角形"""

```python
reasoning_modules_str = "\n".join(reasoning_modules)
```

```python
overall_chain.invoke(
    {"task_description": task_example, "reasoning_modules": reasoning_modules_str}
)
```

```python

```

```python

```

”
将以上内容翻译成中文