以下是您提供的文本的中文翻译：

```
# Wikibase Agent

这个笔记本演示了一个非常简单的wikibase代理，它使用sparql生成。除非您修改以下代码以使用其他LLM提供者，否则这个代理 intend to work against any wikibase instance, 我们会从http://wikidata.org开始测试。

如果您对wikibase和sparql感兴趣，请考虑帮助改进这个代理。更多关于此代理的信息和开放问题可以在[这里](https://github.com/donaldziff/langchain-wikibase)找到。


## 前提条件

### API密钥和其他秘密

我们使用一个`.ini`文件，像这样：
```
[OPENAI]
OPENAI_API_KEY=xyzzy
[WIKIDATA]
WIKIDATA_USER_AGENT_HEADER=argle-bargle
```


```python
import configparser

config = configparser.ConfigParser()
config.read("./secrets.ini")
```


    ['./secrets.ini']



### OpenAI API密钥

除非您修改代码以下部分以使用其他LLM提供者，否则需要一个OpenAI API密钥。


```python
openai_api_key = config["OPENAI"]["OPENAI_API_KEY"]
import os

os.environ.update({"OPENAI_API_KEY": openai_api_key})
```

### Wikidata用户代理头

Wikidata政策要求用户代理头。参见https://meta.wikimedia.org/wiki/User-Agent_policy。然而，目前这个政策并没有得到严格执行。


```python
wikidata_user_agent_header = (
    None
    if not config.has_section("WIKIDATA")
    else config["WIKIDATA"]["WIKIDATA_USER_AGENT_HEADER"]
)
```

### 启用跟踪（如果需要）


```python
# import os
# os.environ["LANGCHAIN_HANDLER"] = "langchain"
# os.environ["LANGCHAIN_SESSION"] = "default" # 确保这个会话实际上存在。
```

# 工具

提供了三个工具供这个简单代理使用：
* `ItemLookup`：用于查找项目的q编号
* `PropertyLookup`：用于查找属性的p编号
* `SparqlQueryRunner`：用于运行sparql查询

## 项目和属性查找

项目和属性查找实现在一个单一的方法中，使用一个elastic search端点。并非所有的wikibase实例都有它，但我们将从wikidata开始。


```python
def get_nested_value(o: dict, path: list) -> any:
    current = o
    for key in path:
        try:
            current = current[key]
        except KeyError:
            return None
    return current


from typing import Optional

import requests


def vocab_lookup(
    search: str,
    entity_type: str = "item",
    url: str = "https://www.wikidata.org/w/api.php",
    user_agent_header: str = wikidata_user_agent_header,
    srqiprofile: str = None,
) -> Optional[str]:
    headers = {"Accept": "application/json"}
    if wikidata_user_agent_header is not None:
        headers["User-Agent"] = wikidata_user_agent_header

    if entity_type == "item":
        srnamespace = 0
        srqiprofile = "classic_noboostlinks" if srqiprofile is None else srqiprofile
    elif entity_type == "property":
        srnamespace = 120
        srqiprofile = "classic" if srqiprofile is None else srqiprofile
    else:
        raise ValueError("entity_type必须为'property'或'item'")

    params = {
        "action": "query",
        "list": "search",
        "srsearch": search,
        "srnamespace":