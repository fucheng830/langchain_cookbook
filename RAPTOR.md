RAPTOR：树状组织检索的递归抽象处理

RAPTOR论文介绍了一种有趣的文档索引和检索方法：

* 叶子是起始文档的一组
* 叶子被嵌入和聚类
* 然后将聚类总结为跨类似文档的高级（更抽象）信息汇总

这个过程是递归进行的，结果是从原始文档（叶子）到更抽象摘要的“树”。

我们可以应用这个方法在不同的尺度上；叶子可以是：

* 单个文档中的文本块（如论文中所示）
* 完整文档（如下所示）

随着更长的上下文LLM，有可能对完整文档执行此操作。

![Screenshot 2024-03-04 at 12.45.25 PM.png](72039e0c-e8c4-4b17-8780-04ad9fc584f3.png)

### 文档

让我们将这个应用到LangChain的LCEL文档。

在这种情况下，每个文档都是LCEL文档中的一个唯一网页。

上下文从< 2k令牌到> 10k令牌不等。

```python
import matplotlib.pyplot as plt
import tiktoken
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """返回文本字符串中的令牌数。"""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# LCEL文档
url = "https://python.langchain.com/docs/expression_language/"
loader = RecursiveUrlLoader(
    url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

# LCEL w/ PydanticOutputParser（在主要LCEL文档之外）
url = "https://python.langchain.com/docs/modules/model_io/output_parsers/quick_start"
loader = RecursiveUrlLoader(
    url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
)
docs_pydantic = loader.load()

# LCEL w/ Self Query（在主要LCEL文档之外）
url = "https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/"
loader = RecursiveUrlLoader(
    url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
)
docs_sq = loader.load()

# 文档文本
docs.extend([*docs_pydantic, *docs_sq])
docs_texts = [d.page_content for d in docs]

# 计算每个文档的令牌数
counts = [num_tokens_from_string(d, "cl100k_base") for d in docs_texts]

# 绘制令牌计数的直方图
plt.figure(figsize=(10, 6))
plt.hist(counts, bins=30, color="blue", edgecolor="black", alpha=0.7)
plt.title("令牌计数的直方图")
plt.xlabel("令牌数")
plt.ylabel("频率")
plt.grid(axis="y", alpha=0.75)

# 显示直方图
plt.show
```

```python
# 文档文本拼接
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)
print(
    "所有上下文中的令牌数：%s"
    % num_tokens_from_string(concatenated_content, "cl100k_