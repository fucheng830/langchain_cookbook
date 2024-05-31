“# 计划与执行

计划与执行代理通过首先规划要做什么，然后执行子任务来达成目标。这一理念主要受到[BabyAGI](https://github.com/yoheinakajima/babyagi)和[“计划与解决”论文](https://arxiv.org/abs/2305.04091)的启发。

规划几乎总是由大型语言模型（LLM）完成。

执行通常由配备工具的独立代理完成。

## 导入

```python
from langchain.chains import LLMMathChain
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain_openai import ChatOpenAI, OpenAI
```

## 工具

```python
search = DuckDuckGoSearchAPIWrapper()
llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name="搜索",
        func=search.run,
        description="当你需要回答关于当前事件的问题时很有用",
    ),
    Tool(
        name="计算器",
        func=llm_math_chain.run,
        description="当你需要回答数学问题时很有用",
    ),
]
```

## 规划者、执行者和代理

```python
model = ChatOpenAI(temperature=0)
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor)
```

## 运行示例

```python
agent.run(
    "英国现任首相是谁？他们的当前年龄的0.43次方是多少？"
)
```

    
    
    [1m> 进入新的AgentExecutor链...[0m
    [32;1m[1;3m行动：
    {
      "action": "搜索",
      "action_input": "英国现任首相"
    }[0m
    
    [1m> 链结束。[0m
    
    
    [1m> 进入新的AgentExecutor链...[0m
    [32;1m[1;3m行动：
    ```
    {
      "action": "搜索",
      "action_input": "英国现任首相"
    }
    ```[0m
    观察： [36;1m[1;3m右下角：里希·苏纳克是现任首相，也是首位非白人首相。英国首相是陛下政府的最高部长，也是英国内阁的首脑。3分钟。英国首相里希·苏纳克在周三的一次演讲中阐述了他对性别身份的立场，称“男人是男人，女人是女人”是“常识”...前财政大臣里希·苏纳克是英国的新首相。以下是你需要了解的关于他的信息。他在今年第二次竞选后获胜，在九月份输给了利兹·特拉斯...以色列总理本雅明·内塔尼亚胡周三与美国总统乔·拜登通了电话，总理办公室在一份声明中表示。内塔尼亚胡“感谢总统对...的强烈言论”...由Yasmeen Serhan /伦敦更新：2022年10月25日12:56 PM EDT |最初发布：2022年10月24日9:17 AM EDT 如果你听过这个故事，请阻止我：在政治动荡之后...[0m
    思考：[32;1m[1;3m搜索结果表明里希·苏纳克是英国现任首相。然而，重要的是要注意这些信息可能不准确或不是最新的。[0m
    
    [1m> 链结束。[0m
    
    
    [1m> 进入新的AgentExecutor链...[0m
    [32;1m[1;3m行动：
    ```
    {
      "action": "搜索",
      "action_input": "英国首相的当前年龄"
    }
    ```[0m
    观察： [36;1m[1;3m里希·苏纳克多大了？苏纳克先生出生于1980年5月12日，现年42岁。他于2015年首次成为议员，时年34岁，自那时起一直代表约克郡的里士满选区。他...首相上任时的年龄从最年长到最年轻，首相的年龄如下：温斯顿·丘吉尔 - 65岁 詹姆斯·卡拉汉 - 64岁 克莱门特·艾德礼 - 62岁...安娜·考夫曼 今日美国 在利兹·特拉斯辞去首相职务几天后，英国有了新首相。特拉斯在任仅45天，将被里希...广告 里希·苏纳克是现代英国最年轻的首相。苏纳克先生42岁，2015年开始进入议会。里希·苏纳克被任命为财政大臣...现任英国首相是现代英国的首位首相，当时英国在1922年（当时26个爱尔兰郡脱离并创建了爱尔兰自由州）有效成立，尽管国家直到1927年（斯坦利·鲍德温担任首相时）才正式更名。[11][0m
    思考：[32;1m[1;3m根据搜索结果，里希·苏纳克似乎是英国现任首相。然而，我找不到关于他年龄的具体信息。您是否希望我再次搜索现任首相的当前年龄？
    
    行动：
    ```
    {
      "action": "搜索",
      "action_input": "里希·苏纳克的年龄"
    }
    ```[0m
    观察： [36;1m[1;3m里希·苏纳克现年42岁，这使他成为现代英国最年轻的首相。里希·苏纳克多高？里希·苏纳克多大了？里希·苏纳克出生于1980年5月12日，现年42岁。父母和民族 苏纳克的父母在东非的印度裔家庭出生后...出生于1980年5月12日，里希目前42岁。自2015年以来，他一直是议会议员，代表里士满，并在鲍里斯·约翰逊担任首相期间担任过财政大臣和财政大臣等职务。家庭 穆蒂，42岁，是印度亿万富翁NR纳拉亚纳·穆尔蒂的女儿，常被称为印度的比尔·盖茨，他创立了软件公司Infosys。据报道，他的...苏纳克成为首位领导该国的非白人，并且，在42岁时，是超过一个世纪以来最年轻的首相。像大多数政治家一样，苏纳克受到一些人的尊敬...[0m
    思考：[32;1m[1;3m根据搜索结果，里希·苏纳克目前42岁。他出生于1980年5月12日。[0m
    
    [1m> 链结束。[0m
    
    
    [1m> 进入新的AgentExecutor链...[0m
    [32;1m[1;3m思考：要计算年龄的0.43次方，我可以使用计算器工具。
    
    行动：
    ```json
    {
      "action": "计算器",
      "action_input": "42^0.43"
    }
    ```[0m
    
    [1m> 进入新的LLMMathChain链...[0m
    42^0.43[32;1m[1;3m```text
    42**0.43
    ```
    ...numexpr.evaluate("42**0.43")...
    [0m
    答案： [33;1m[1;3m4.9888126515157[0m
    [1m> 链结束。[0m
    
    观察： [33;1m[1;3m答案：4.9888126515157[0m
    思考：[32;1m[1;3m年龄的0.43次方大约是4.9888126515157。
    
    最终答案：
    ```json
    {
      "action": "最终答案",
      "action_input": "年龄的0.43次方大约是4.9888126515157。"
    }
    ```[0m
    
    [1m> 链结束。[0m
    
    
    [1m> 进入新的AgentExecutor链...[0m
    [32;1m[1;3m行动：
    ```
    {
      "action": "最终答案",
      "action_input": "英国现任首相是里希·苏纳克。他的年龄的0.43次方大约是4.9888126515157。"
    }
    ```[0m
    
    [1m> 链结束。[0m





'英国现任首相是里希·苏纳克。他的年龄的0.43次方大约是4.9888126515157。'




```python

```

”