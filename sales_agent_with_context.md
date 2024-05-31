"
# SalesGPT - 具有知识库和生成Stripe支付链接能力的上下文感知AI销售助理

这个笔记本展示了实现一个**上下文感知**AI销售代理的实现，该代理具有产品知识库，并且能够实际完成销售。

这个笔记本最初是在[filipmichalsky/SalesGPT](https://github.com/filip-michalsky/SalesGPT)由[@FilipMichalsky](https://twitter.com/FilipMichalsky)发布的。

SalesGPT是上下文感知的，这意味着它可以理解销售对话中的哪个部分，并相应地采取行动。

因此，这个笔记本演示了如何使用AI来自动化销售发展代表的活动，例如外展销售电话。

此外，AI销售代理可以访问工具，这些工具允许它与其他系统交互。

在这里，我们展示了AI销售代理如何使用一个**产品知识库**来谈论特定公司的产品，
因此增加了相关性并减少了虚构。

此外，我们展示了我们的AI销售代理如何**通过与名为[Mindware](https://www.mindware.co/)的AI代理高速公路的集成**来**生成销售**。在实践中，这允许代理自主为您的客户**通过Stripe支付您的产品**生成支付链接。

在这个实现中，我们利用了[`langchain`](https://github.com/hwchase17/langchain)库，特别是[自定义代理配置](https://langchain-langchain.vercel.app/docs/modules/agents/how_to/custom_agent_with_tool_retrieval)并受到[BabyAGI](https://github.com/yoheinakajima/babyagi)架构的启发。

## 导入库并设置您的环境

```python
import os
import re

# 确保您在本地保存了.env文件，其中包含您的API密钥
from dotenv import load_dotenv

load_dotenv()

from typing import Any, Callable, Dict, List, Union

from langchain.agents import AgentExecutor, LLMSingleActionAgent, Tool
from langchain.agents.agent import AgentOutputParser
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.prompts.base import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
```

### SalesGPT架构

1. 播种SalesGPT代理
2. 运行销售代理以决定要做什么：

    a) 使用工具，例如在知识库中查找产品信息或生成支付链接
    
    b) 输出对用户的响应
3. 运行销售阶段识别代理以识别销售代理处于哪个阶段，并相应地调整其行为。

这里是架构的示意图：

### 架构图

<img src="https://demo-bucket-45.s3.amazonaws.com/new_flow2.png"  width="800" height="440">


### 销售对话阶段。

代理使用一个助手来保持其处于对话的哪个阶段。这些阶段是由ChatGPT生成的，可以轻松修改以适应其他用例或对话模式。

1. 介绍：通过介绍自己和公司开始对话。在保持专业语调的同时，要礼貌和尊重。

2. 资格确认：通过确认潜在客户是否是您产品/服务的正确人选来确认他们。确保他们有购买决策的权力。

3. 价值主张：简要解释您的产品/服务如何使潜在客户受益。关注您的产品/服务的独特卖点和价值主张，这些是将其与竞争对手区分开来的。

4. 需求分析：通过提出开放式问题来发现潜在客户的需求和痛点。仔细倾听他们的回答并做笔记。

5. 解决方案展示：根据潜在客户的需求，将您的产品/服务作为解决他们痛点的问题的解决方案进行展示。

6. 反对处理：解决潜在客户对您的产品/服务的任何反对意见。准备好提供证据或推荐信来支持您的说法