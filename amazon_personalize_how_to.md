“# 亚马逊个性化推荐

[亚马逊个性化推荐](https://docs.aws.amazon.com/personalize/latest/dg/what-is-personalize.html)是一项完全托管的机器学习服务，它利用您的数据为用户生成物品推荐。它还能根据用户对特定物品或物品元数据的亲和力生成用户分段。

本笔记本将介绍如何使用亚马逊个性化推荐链。在开始使用下面的笔记本之前，您需要一个亚马逊个性化推荐活动的ARN或推荐者的ARN。

以下是一个[教程](https://github.com/aws-samples/retail-demo-store/blob/master/workshop/1-Personalization/Lab-1-Introduction-and-data-preparation.ipynb)，用于在亚马逊个性化推荐上设置campaign_arn/recommender_arn。一旦设置了campaign_arn/recommender_arn，您就可以在langchain生态系统中使用它。

## 1. 安装依赖

```python
!pip install boto3
```

## 2. 示例用例

### 2.1 [用例-1] 设置亚马逊个性化推荐客户端并获取推荐

```python
from langchain_experimental.recommenders import AmazonPersonalize

recommender_arn = "<插入ARN>"

client = AmazonPersonalize(
    credentials_profile_name="default",
    region_name="us-west-2",
    recommender_arn=recommender_arn,
)
client.get_recommendations(user_id="1")
```

### 2.2 [用例-2] 使用个性化推荐链总结结果

```python
from langchain.llms.bedrock import Bedrock
from langchain_experimental.recommenders import AmazonPersonalizeChain

bedrock_llm = Bedrock(model_id="anthropic.claude-v2", region_name="us-west-2")

# 创建个性化推荐链
# 如果您不想要总结，请使用return_direct=True
chain = AmazonPersonalizeChain.from_llm(
    llm=bedrock_llm, client=client, return_direct=False
)
response = chain({"user_id": "1"})
print(response)
```

### 2.3 [用例-3] 使用自定义提示调用亚马逊个性化推荐链

```python
from langchain.prompts.prompt import PromptTemplate

RANDOM_PROMPT_QUERY = """
您是一位熟练的公关人员。根据以下电影和用户信息，撰写一封高转化率的营销邮件，宣传下周在视频点播流媒体平台上线的几部电影。您的邮件将利用讲故事和说服性语言的力量。
    推荐的电影《movie》标签中包含所有必须推荐的电影。给出电影的概要以及人们应该观看的原因。
    将邮件放在<email>标签之间。

    <movie>
    {result}
    </movie>

    助手:
    """

RANDOM_PROMPT = PromptTemplate(input_variables=["result"], template=RANDOM_PROMPT_QUERY)

chain = AmazonPersonalizeChain.from_llm(
    llm=bedrock_llm, client=client, return_direct=False, prompt_template=RANDOM_PROMPT
)
chain.run({"user_id": "1", "item_id": "234"})
```

### 2.4 [用例-4] 在顺序链中调用亚马逊个性化推荐

```python
from langchain.chains import LLMChain, SequentialChain

RANDOM_PROMPT_QUERY_2 = """
您是一位熟练的公关人员。根据以下电影和用户信息，撰写一封高转化率的营销邮件，宣传下周在视频点播流媒体平台上线的几部电影。您的邮件将利用讲故事和说服性语言的力量。
    您希望邮件能给用户留下深刻印象，因此要让它对他们有吸引力。
    推荐的电影《movie》标签中包含所有必须推荐的电影。给出电影的概要以及人们应该观看的原因。
    将邮件放在<email>标签之间。

    <movie>
    {result}
    </movie>

    助手:
    """

RANDOM_PROMPT_2 = PromptTemplate(
    input_variables=["result"], template=RANDOM_PROMPT_QUERY_2
)
personalize_chain_instance = AmazonPersonalizeChain.from_llm(
    llm=bedrock_llm, client=client, return_direct=True
)
random_chain_instance = LLMChain(llm=bedrock_llm, prompt=RANDOM_PROMPT_2)
overall_chain = SequentialChain(
    chains=[personalize_chain_instance, random_chain_instance],
    input_variables=["user_id"],
    verbose=True,
)
overall_chain.run({"user_id": "1", "item_id": "234"})
```

### 2.5 [用例-5] 调用亚马逊个性化推荐并获取元数据

```python
recommender_arn = "<插入ARN>"
metadata_column_names = [
    "<插入元数据列名-1>",
    "<插入元数据列名-2>",
]
metadataMap = {"ITEMS": metadata_column_names}

client = AmazonPersonalize(
    credentials_profile_name="default",
    region_name="us-west-2",
    recommender_arn=recommender_arn,
)
client.get_recommendations(user_id="1", metadataColumns=metadataMap)
```

### 2.6 [用例-6] 使用返回的元数据总结个性化推荐链的结果

```python
bedrock_llm = Bedrock(model_id="anthropic.claude-v2", region_name="us-west-2")

# 创建个性化推荐链
# 如果您不想要总结，请使用return_direct=True
chain = AmazonPersonalizeChain.from_llm(
    llm=bedrock_llm, client=client, return_direct=False
)
response = chain({"user_id": "1", "metadata_columns": metadataMap})
print(response)
```

”