半结构化RAG

许多文档包含文本和表格的混合内容。

半结构化数据对传统RAG至少有两个挑战：

* 分割文本可能会打断表格，导致检索中的数据损坏
* 嵌入表格可能会给语义相似性搜索带来挑战

本食谱展示了如何对具有半结构化数据的文档执行RAG：

* 我们将使用[Unstructured](https://unstructured.io/)解析文本和表格（PDFs）。
* 我们将使用[多向量检索器](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector)以更好地存储原始表格、文本和表摘要，适用于检索。
* 我们将使用[LCEL](https://python.langchain.com/docs/expression_language/)实现所用的链。

整体流程如下：

![MVR.png](7b5c5a30-393c-4b27-8fa1-688306ef2aef.png)

## 包

```python
! pip install langchain unstructured[all-docs] pydantic lxml langchainhub
```

Unstructured用于PDF分区的分区工具将使用：

* `tesseract`进行光学字符识别（OCR）
* `poppler`进行PDF渲染和处理

```python
! brew install tesseract
! brew install poppler
```

## 数据加载

### 分区PDF文本和表格

将[`LLaMA2`](https://arxiv.org/pdf/2307.09288.pdf)论文应用于分区PDF文档，使用Unstructured的[`partition_pdf`](https://unstructured-io.github.io/unstructured/core/partition.html#partition-pdf)，它通过使用布局模型提取元素，如表格。

我们还可以使用`Unstructured`分块，它：

* 尝试识别文档部分（例如，引言等）
* 然后，建立维护部分的同时也尊重用户定义的分块大小的文本块

```python
path = "/Users/rlm/Desktop/Papers/LLaMA2/"
```

```python
from typing import Any

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

# 获取元素
raw_pdf_elements = partition_pdf(
    filename=path + "LLaMA2.pdf",
    # Unstructured首先查找嵌入的图像块
    extract_images_in_pdf=False,
    # 使用布局模型（YOLOX）获取边界框（对于表格）并查找标题
    # 标题是文档的任何子部分
    infer_table_structure=True,
    # 后处理以聚合文本一旦我们有了标题
    chunking_strategy="by_title",
    # 分块参数以聚合文本块
    # 在尝试创建新块3800字符后
    # 尝试保持块> 2000字符
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path,
)
```

我们可以检查`partition_pdf`提取的元素。

`CompositeElement`是聚合的块。

```python
# 创建一个字典来存储每种类型的计数
category_counts = {}

for element in raw_pdf_elements:
    category = str(type(element))
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1

# 唯一类别将具有独特元素
unique_categories = set(category_counts.keys())
category_counts
```

```
{<class 'unstructured.documents.elements.CompositeElement'>: 184,
 <class 'unstructured.documents.elements.Table'>: 47,
 <class 'unstructured.documents.elements.TableChunk'>: 2}
```

```python
class Element(BaseModel):
    type: