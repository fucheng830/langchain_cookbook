"
# 使用HyDE改进文档索引
这个笔记本介绍了如何使用假设性文档嵌入（HyDE），具体内容在[这篇论文](https://arxiv.org/abs/2212.10496)中有详细描述。

在较高层面上，HyDE是一种嵌入技术，它接收查询，生成一个假设的答案，然后将这个生成的文档嵌入，并将其作为最终的示例。

为了使用HyDE，我们因此需要提供一个基础嵌入模型，以及一个可以用来生成文档的LLMChain。默认情况下，HyDE类带有某些默认提示（详见论文），但我们也可以创建自己的。

```python
from langchain.chains import HypotheticalDocumentEmbedder, LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings
```


```python
base_embeddings = OpenAIEmbeddings()
llm = OpenAI()
```




```python
# 使用“web_search”提示加载
embeddings = HypotheticalDocumentEmbedder.from_llm(llm, base_embeddings, "web_search")
```


```python
# 现在我们可以像使用任何其他嵌入类一样使用它！
result = embeddings.embed_query("泰姬陵在哪里？")
```

## 多代生成
我们还可以生成多个文档，然后合并这些嵌入。默认情况下，我们是通过取平均值来合并这些的。我们可以通过改变生成文档的LLM使其返回多个内容来实现。

```python
multi_llm = OpenAI(n=4, best_of=4)
```


```python
embeddings = HypotheticalDocumentEmbedder.from_llm(
    multi_llm, base_embeddings, "web_search"
)
```


```python
result = embeddings.embed_query("泰姬陵在哪里？")
```

## 使用自己的提示
除了使用预配置的提示外，我们还可以很容易地构建自己的提示，并在生成文档的LLMChain中使用这些提示。如果我们知道查询将涉及的领域，这将非常有用，因为我们可以使提示生成与该领域更相似的文本。

以下示例，让我们使其生成关于国情咨文的文本（因为我们将在下一个示例中使用它）。

```python
prompt_template = """请回答用户关于最近一次国情咨文的提问
问题：{question}
答案："""
prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
llm_chain = LLMChain(llm=llm, prompt=prompt)
```


```python
embeddings = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain, base_embeddings=base_embeddings
)
```


```python
result = embeddings.embed_query(
    "总统在国情咨文中关于Ketanji Brown Jackson说了什么"
)
```

## 使用HyDE
现在我们有了HyDE，我们可以像使用任何其他嵌入类一样使用它！以下是在国情咨文示例中找到类似段落的用法。

```python
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

with open("../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)
```


```python
docsearch = Chroma.from_texts(texts, embeddings)

query = "总统在国情咨文中关于Ketanji Brown Jackson说了什么"
docs = docsearch.similarity_search(query)
```

    使用直接本地API运行Chroma。
    使用DuckDB内存数据库。数据将是瞬时的。



```python
print(docs[0].page_content)
```

    在各个州，新的法律已经被通过，不仅是为了压制选票，而是为了颠覆整个选举。
    
    我们不能让这种情况发生。
    
    今晚。我要求参