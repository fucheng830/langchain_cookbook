“# 人类输入LLM

类似于模拟的LLM，LangChain提供了一个伪LLM类，可用于测试、调试或教育目的。这允许你模拟对LLM的调用，并模拟如果人类收到这些提示会如何响应。

在本笔记本中，我们将介绍如何使用这个功能。

我们首先使用HumanInputLLM在一个代理中。

```python
from langchain_community.llms.human import HumanInputLLM
```

```python
from langchain.agents import AgentType, initialize_agent, load_tools
```

由于本笔记本中我们将使用`WikipediaQueryRun`工具，如果你还没有安装`wikipedia`包，你可能需要安装它。

```python
%pip install wikipedia
```

```python
tools = load_tools(["wikipedia"])
llm = HumanInputLLM(
    prompt_func=lambda prompt: print(
        f"\n===PROMPT====\n{prompt}\n=====END OF PROMPT======"
    )
)
```

```python
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
```

```python
agent.run("'Bocchi the Rock!'是什么？")
```

```

    [1m> 进入新的AgentExecutor链...[0m
    
    ===PROMPT====
    尽你所能回答以下问题。你拥有以下工具：
    
    Wikipedia: 一个围绕Wikipedia的包装器。当你需要回答关于人物、地点、公司、历史事件或其他主题的一般问题时，这个工具很有用。输入应是一个搜索查询。
    
    使用以下格式：
    
    问题：你必须回答的输入问题
    思考：你应该始终思考要做什么
    行动：采取的行动，应是[Wikipedia]之一
    行动输入：行动的输入
    观察：行动的结果
    ...（此思考/行动/行动输入/观察可以重复N次）
    思考：我现在知道了最终答案
    最终答案：原始输入问题的最终答案
    
    开始！
    
    问题：'Bocchi the Rock!'是什么？
    思考：
    =====END OF PROMPT======
    [32;1m[1;3m我需要使用一个工具。
    行动：Wikipedia
    行动输入：Bocchi the Rock!，日本四格漫画和动画系列。[0m
    观察： [36;1m[1;3m页面：Bocchi the Rock!
    摘要：Bocchi the Rock!（ぼっち・ざ・ろっく!，Bocchi Za Rokku!）是由浜地亜季编写和插图的日本四格漫画系列。自2017年12月起在Houbunsha的青年漫画杂志Manga Time Kirara Max上连载。截至2022年11月，其章节已收录在五卷单行本中。
    由CloverWorks制作的动画电视系列改编版于2022年10月至12月播出。该系列因其写作、喜剧、角色和社交焦虑的描绘而受到赞誉，动画的视觉创造力也受到好评。
    
    页面：Manga Time Kirara
    摘要：Manga Time Kirara（まんがタイムきらら，Manga Taimu Kirara）是由Houbunsha出版的日本青年漫画杂志，主要连载四格漫画。该杂志每月9日发售，最初于2002年5月17日作为另一本Houbunsha杂志Manga Time的特刊首次出版。该杂志的角色出现在名为Kirara Fantasia的跨界角色扮演游戏中。
    
    页面：Manga Time Kirara Max
    摘要：Manga Time Kirara Max（まんがタイムきららMAX）是由Houbunsha出版的日本四格青年漫画杂志。它是“Kirara”系列中的第三本杂志，继“Manga Time Kirara”和“Manga Time Kirara Carat”之后。第一期于2004年9月29日发行。目前该杂志每月19日发行。
    思考：
    ===PROMPT====
    尽你所能回答以下问题。你拥有以下工具：
    
    Wikipedia: 一个围绕Wikipedia的包装器。当你需要回答关于人物、地点、公司、历史事件或其他主题的一般问题时，这个工具很有用。输入应是一个搜索查询。
    
    使用以下格式：
    
    问题：你必须回答的输入问题
    思考：你应该始终思考要做什么
    行动：采取的行动，应是[Wikipedia]之一
    行动输入：行动的输入
    观察：行动的结果
    ...（此思考/行动/行动输入/观察可以重复N次）
    思考：我现在知道了最终答案
    最终答案：原始输入问题的最终答案
    
    开始！
    
    问题：'Bocchi the Rock!'是什么？
    思考：我需要使用一个工具。
    行动：Wikipedia
    行动输入：Bocchi the Rock!，日本四格漫画和动画系列。
    观察：页面：Bocchi the Rock!
    摘要：Bocchi the Rock!（ぼっち・ざ・ろっく!，Bocchi Za Rokku!）是由浜地亜季编写和插图的日本四格漫画系列。自2017年12月起在Houbunsha的青年漫画杂志Manga Time Kirara Max上连载。截至2022年11月，其章节已收录在五卷单行本中。
    由CloverWorks制作的动画电视系列改编版于2022年10月至12月播出。该系列因其写作、喜剧、角色和社交焦虑的描绘而受到赞誉，动画的视觉创造力也受到好评。
    
    页面：Manga Time Kirara
    摘要：Manga Time Kirara（まんがタイムきらら，Manga Taimu Kirara）是由Houbunsha出版的日本青年漫画杂志，主要连载四格漫画。该杂志每月9日发售，最初于2002年5月17日作为另一本Houbunsha杂志Manga Time的特刊首次出版。该杂志的角色出现在名为Kirara Fantasia的跨界角色扮演游戏中。
    
    页面：Manga Time Kirara Max
    摘要：Manga Time Kirara Max（まんがタイムきららMAX）是由Houbunsha出版的日本四格青年漫画杂志。它是“Kirara”系列中的第三本杂志，继“Manga Time Kirara”和“Manga Time Kirara Carat”之后。第一期于2004年9月29日发行。目前该杂志每月19日发行。
    思考：这些不是相关文章。
    行动：Wikipedia
    行动输入：Bocchi the Rock!，由浜地亜季编写和插图的日本四格漫画系列。
    观察：页面：Bocchi the Rock!
    摘要：Bocchi the Rock!（ぼっち・ざ・ろっく!，Bocchi Za Rokku!）是由浜地亜季编写和插图的日本四格漫画系列。自2017年12月起在Houbunsha的青年漫画杂志Manga Time Kirara Max上连载。截至2022年11月，其章节已收录在五卷单行本中。
    由CloverWorks制作的动画电视系列改编版于2022年10月至12月播出。该系列因其写作、喜剧、角色和社交焦虑的描绘而受到赞誉，动画的视觉创造力也受到好评。
    思考：
    =====END OF PROMPT======
    [32;1m[1;3m成功了。
    最终答案：Bocchi the Rock!是一个四格漫画系列和动画电视系列。该系列因其写作、喜剧、角色和社交焦虑的描绘而受到赞誉，动画的视觉创造力也受到好评。[0m
    
    [1m> 链完成。[0m





    "Bocchi the Rock!是一个四格漫画系列和动画电视系列。该系列因其写作、喜剧、角色和社交焦虑的描绘而受到赞誉，动画的视觉创造力也受到好评。"




”