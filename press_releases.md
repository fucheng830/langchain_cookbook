“新闻稿数据
=

新闻稿数据由 [Kay.ai](https://kay.ai) 提供。

>新闻稿是公司用来宣布重要事项的工具，包括产品发布、财务业绩报告、合作伙伴关系以及其他重大新闻。分析师广泛使用新闻稿来追踪公司的战略、运营更新和财务表现。
Kay.ai 从多种来源获取所有美国上市公司的新闻稿，这些来源包括公司的官方新闻室以及与各种数据 API 提供商的合作。
此数据更新至9月30日，免费访问，如需实时数据流，请联系我们 at hello@kay.ai 或 [在推特上联系我们](https://twitter.com/vishalrohra_)。

设置
=

首先，您需要安装 `kay` 包。您还需要一个 API 密钥：您可以在 [https://kay.ai](https://kay.ai/) 免费获取一个。获得 API 密钥后，您必须将其设置为环境变量 `KAY_API_KEY`。

在本例中，我们将使用 `KayAiRetriever`。请查看 [kay 笔记本](/docs/integrations/retrievers/kay) 以获取有关其接受的参数的更详细信息。

示例
=

```python
# 设置 Kay 和 OpenAI 的 API 密钥
from getpass import getpass

KAY_API_KEY = getpass()
OPENAI_API_KEY = getpass()
```

     ········
     ········



```python
import os

os.environ["KAY_API_KEY"] = KAY_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
```


```python
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import KayAiRetriever
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")
retriever = KayAiRetriever.create(
    dataset_id="company", data_types=["PressRelease"], num_contexts=6
)
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
```


```python
# 更多示例问题在 https://kay.ai 的 Playground 中
questions = [
    "医疗保健行业如何采用生成式AI工具？",
    # "可再生能源领域最近面临哪些挑战？",
]
chat_history = []

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    print(f"-> **问题**: {question} \n")
    print(f"**答案**: {result['answer']} \n")
```

    -> **问题**: 医疗保健行业如何采用生成式AI工具？ 
    
    **答案**: 医疗保健行业正在采用生成式AI工具来改善患者护理和行政任务的各个方面。像HCA Healthcare Inc、Amazon Com Inc和Mayo Clinic这样的公司已经与Google Cloud、AWS和Microsoft等技术提供商合作，实施生成式AI解决方案。
    
    HCA Healthcare正在测试一种护士交接工具，该工具能快速准确地生成草稿报告，护士们对此表现出兴趣。他们还在探索使用Google的医学调优Med-PaLM 2 LLM来支持护理人员提出复杂的医学问题。
    
    Amazon Web Services (AWS) 推出了AWS HealthScribe，这是一种生成式AI驱动的服务，可自动创建临床文档。然而，将多个AI系统整合成一个统一的解决方案需要大量的工程资源，包括访问AI专家、医疗数据和计算能力。
    
    Mayo Clinic是首批部署Microsoft 365 Copilot的医疗保健组织之一，这是一种结合了Microsoft 365组织数据的大型语言模型的生成式AI服务。该工具有可能自动化任务，如表格填写，减轻医疗保健提供者的行政负担，使他们能够更多地关注患者护理。
    
    总体而言，医疗保健行业正在认识到生成式AI工具在提高效率、自动化任务和增强患者护理方面的潜在好处。 
    



”