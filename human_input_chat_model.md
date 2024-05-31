```
# 人类输入聊天模型

LangChain 还提供了一个伪聊天模型类，可以用于测试、调试或教育目的。这允许您模拟人类如何回应如果他们收到了这些信息。

在下面的笔记本中，我们将介绍如何使用这个人类输入聊天模型。

我们从在代理中使用 HumanInputChatModel 开始。

```python
from langchain_community.chat_models.human import HumanInputChatModel
```

由于我们将使用 `WikipediaQueryRun` 工具，如果您还没有安装 `wikipedia` 包，您可能需要安装它。

```python
%pip install wikipedia
```

    /Users/mskim58/dev/research/chatbot/github/langchain/.venv/bin/python: No module named pip
    Note: you may need to restart the kernel to use updated packages.



```python
from langchain.agents import AgentType, initialize_agent, load_tools
```


```python
tools = load_tools(["wikipedia"])
llm = HumanInputChatModel()
```


```python
agent = initialize_agent(
    tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
```


```python
agent("What is Bocchi the Rock?")
```

    
    
    [1m> Entering new  chain...[0m
    
     ======= start of message ======= 
    
    
    type: system
    data:
      content: "Answer the following questions as best you can. You have access to the following tools:\n\nWikipedia: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.\n\nThe way you use the tools is by specifying a json blob.\nSpecifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).\n\nThe only values that should be in the \"action\" field are: Wikipedia\n\nThe $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:\n\n```\n{\n  \"action\": $TOOL_NAME,\n  \"action_input\": $INPUT\n}\n```\n\nALWAYS use the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction:\n```\n$JSON_BLOB\n```\nObservation: the result of the action\n... (this Thought/Action/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin! Reminder to always use the exact characters `Final Answer` when responding."
      additional_kwargs: {}
    
    ======= end of message ======= 
    
    
    
     ======= start of message ======= 
    
    
    type: human
    data:
      content: 'What is Bocchi the Rock?
    
    
        '
      additional_kwargs: {}
      example: false
    
    ======= end of message ======= 
    
    
    [32;1m[1;3mAction:
    ```
    {
      "action": "Wikipedia",
      "action_input": "What is Bocchi the Rock?"
    }
    ```[0m
    Observation: [36;1m[1;3mPage: Bocchi the Rock!\nSummary: Bocchi the Rock! (ぼっち・ざ・ろっく!, Botchi Za Rokku!) is a Japanese four-panel manga series written and illustrated by Aki Hamaji. It has been serialized in Houbunsha's seinen manga magazine Manga Time Kirara Max since December 2017. Its chapters have been collected in five tankōbon volumes as of November 2022.\nAn