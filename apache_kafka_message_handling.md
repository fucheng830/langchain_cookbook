# 使用Apache Kafka路由聊天消息

这个笔记本向你展示了如何使用LangChain的标准聊天功能，同时通过Apache Kafka将聊天消息来回传递。

这个目标是为了模拟一个架构，其中聊天前端和运行作为单独服务的LLM需要通过内部网络进行通信。

这是一种替代典型模式的方法，即请求API从外部服务生成响应（关于为什么这样做，我们在笔记本的最后会有更多详细信息）。

### 1. 安装主要依赖项

依赖项包括：

- Quix Streams库，用于管理与Apache Kafka（或类似工具如Redpanda）的交互，以Pandas-like方式操作。
- LangChain库，用于管理与LLM的交互并存储会话状态。

```python
!pip install quixstreams==2.1.2a langchain==0.0.340 huggingface_hub==0.19.4 langchain-experimental==0.0.42 python-dotenv
```

### 2. 构建并安装llama-cpp-python库（启用CUDA以利用Google Colab GPU）

`llama-cpp-python`库是一个Python包装器，围绕`llama-cpp`库，它允许你有效地利用CPU运行量化LLMs。

当你使用标准的`pip install llama-cpp-python`命令时，你不会得到GPU支持。如果在Google Colab中仅依赖CPU，生成可能会非常慢。以下命令添加了一个额外的选项来构建和安装带有GPU支持的`llama-cpp-python`（确保你在Google Colab中选择了GPU支持的运行时）。

```python
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

### 3. 下载并设置Kafka和Zookeeper实例

从Apache网站下载Kafka二进制文件，并以守护进程方式启动Zookeeper和Kafka服务器。我们使用Apache Kafka的默认配置（由Apache Kafka提供）来启动这些实例。

```python
!curl -sSOL https://dlcdn.apache.org/kafka/3.6.1/kafka_2.13-3.6.1.tgz
!tar -xzf kafka_2.13-3.6.1.tgz
```

```python
!./kafka_2.13-3.6.1/bin/zookeeper-server-start.sh -daemon ./kafka_2.13-3.6.1/config/zookeeper.properties
!./kafka_2.13-3.6.1/bin/kafka-server-start.sh -daemon ./kafka_2.13-3.6.1/config/server.properties
!echo "Waiting for 10 secs until kafka and zookeeper services are up and running"
!sleep 10
```

### 4. 检查Kafka守护进程是否正在运行

显示正在运行的进程，并筛选出Java进程（你应该看到两个—分别是Zookeeper和Kafka的服务器）。

```python
!ps aux | grep -E '[j]ava'
```

### 5. 导入所需依赖项并初始化变量

导入Quix Streams库以与Kafka交互，并导入LangChain组件以运行`ConversationChain`。

```python
# 导入实用库
import json
import random
import re
import time
import uuid
from os import environ
from pathlib import Path
from random import choice, randint, random

from dotenv import load_dotenv

# 导入Hugging Face实用程序以直接从Hugging Face hub下载模型：
from huggingface_hub import hf_hub_download
from langchain.chains import ConversationChain

# 导入Langchain模块以管理提示和会话链：
from langchain.llms import LlamaCpp
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import PromptTemplate, load_prompt
from langchain_core.messages import SystemMessage
from lang