“# HuggingGPT
[HuggingGPT](https://github.com/microsoft/JARVIS)的实现。HuggingGPT是一个系统，用于连接LLM（ChatGPT）与机器学习社区（Hugging Face）。

+ 🔥 论文：https://arxiv.org/abs/2303.17580
+ 🚀 项目：https://github.com/microsoft/JARVIS
+ 🤗 空间：https://huggingface.co/spaces/microsoft/HuggingGPT

## 设置工具

我们设置了[Transformers Agent](https://huggingface.co/docs/transformers/transformers_agents#tools)提供的工具。它包括Transformers支持的工具库以及一些自定义工具，如图像生成器、视频生成器、文本下载器和其他工具。

```python
from transformers import load_tool
```

```python
hf_tools = [
    load_tool(tool_name)
    for tool_name in [
        "文档问答",
        "图像字幕",
        "图像问答",
        "图像分割",
        "语音转文本",
        "摘要",
        "文本分类",
        "文本问答",
        "翻译",
        "huggingface-tools/文本转图像",
        "huggingface-tools/文本转视频",
        "文本转语音",
        "huggingface-tools/文本下载",
        "huggingface-tools/图像变换",
    ]
]
```

## 设置模型和HuggingGPT

我们创建了一个HuggingGPT实例，并使用ChatGPT作为控制器来管理上述工具。

```python
from langchain_experimental.autonomous_agents import HuggingGPT
from langchain_openai import OpenAI

# %env OPENAI_API_BASE=http://localhost:8000/v1
```

```python
llm = OpenAI(model_name="gpt-3.5-turbo")
agent = HuggingGPT(llm, hf_tools)
```

## 运行示例

给定一段文本，展示相关的图像和视频。

```python
agent.run("请给我展示一个关于'一个男孩在跑步'的视频和图片")
```

”