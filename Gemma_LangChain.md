以下是您提供内容的中文翻译：

## 开始使用 LangChain 和 Gemma，无论是在本地还是云端

### 安装依赖项

```python
!pip install --upgrade langchain langchain-google-vertexai
```

### 运行模型

前往 Google Cloud 的 VertexAI Model Garden 控制台：[链接](https://pantheon.corp.google.com/vertex-ai/publishers/google/model-garden/335)，部署您选择的 Gemma 版本。部署完成后，复制其端点编号。

```python
# @title 基本参数
project: str = "PUT_YOUR_PROJECT_ID_HERE"  # @param {type:"string"}
endpoint_id: str = "PUT_YOUR_ENDPOINT_ID_HERE"  # @param {type:"string"}
location: str = "PUT_YOUR_ENDPOINT_LOCAtION_HERE"  # @param {type:"string"}
```

```python
from langchain_google_vertexai import (
    GemmaChatVertexAIModelGarden,
    GemmaVertexAIModelGarden,
)
```

在本地运行 Gemma：

```python
output = llm.invoke("What is the meaning of life?")
print(output)
```

在本地以多轮对话模式运行 Gemma：

```python
from langchain_core.messages import HumanMessage

llm = GemmaChatVertexAIModelGarden(
    endpoint_id=endpoint_id,
    project=project,
    location=location,
)

message1 = HumanMessage(content="How much is 2+2?")
answer1 = llm.invoke([message1])
print(answer1)

message2 = HumanMessage(content="How much is 3+3?")
answer2 = llm.invoke([message1, answer1, message2])

print(answer2)
```

### 从 Kaggle 运行 Gemma

要本地运行 Gemma，您需要从 Kaggle 下载它。为此，您需要登录 Kaggle 平台，创建 API 密钥，并下载 `kaggle.json` 文件。关于 Kaggle 认证的更多信息，请参阅[这里](https://www.kaggle.com/docs/api)。

### 安装

```python
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/kaggle.json
```

在本地以 Kaggle 方式运行 Gemma：

```python
from langchain_google_vertexai import GemmaLocalKaggle
```

在本地以多轮对话模式运行 Gemma：

```python
from langchain_core.messages import HumanMessage

llm = GemmaLocalKaggle(model_name=model_name, keras_backend=keras_backend)
```

### 从 HuggingFace 运行 Gemma

```python
from langchain_google_vertexai import GemmaChatLocalHF, GemmaLocalHF
```

设置 HuggingFace 访问令牌和模型名称：

```python
# @title 基本参数
hf_access_token: str = "PUT_YOUR_TOKEN_HERE"  # @param {type:"string"}
model_name: str = "google/gemma-2b"  # @param {type:"string"}
```

在本地以 HuggingFace 方式运行 Gemma：

```python
llm = GemmaLocalHF(model_name="google/gemma-2b", hf_access_token=hf_access_token)
```

输出结果：

```python
What is the meaning of life?

The question is one of the most important questions in the world.

It’s the question that has
```

请注意，由于篇幅限制，部分内容被截断。如果您需要更多详细信息或有其他问题，请告知。