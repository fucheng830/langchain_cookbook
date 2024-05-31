# Building hotel room search with self-querying retrieval

In this example we'll walk through how to build and iterate on a hotel room search service that leverages an LLM to generate structured filter queries that can then be passed to a vector store.

For an introduction to self-querying retrieval [check out the docs](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query).

## Imports and data prep

In this example we use `ChatOpenAI` for the model and `ElasticsearchStore` for the vector store, but these can be swapped out with an LLM/ChatModel and [any VectorStore that support self-querying](https://python.langchain.com/docs/integrations/retrievers/self_query/).

Download data from: https://www.kaggle.com/datasets/keshavramaiah/hotel-recommendation


```python
!pip install langchain langchain-elasticsearch lark openai elasticsearch pandas
```


```python
import pandas as pd
```


```python
details = (
    pd.read_csv("~/Downloads/archive/Hotel_details.csv")
    .drop_duplicates(subset="hotelid")
    .set_index("hotelid")
)
attributes = pd.read_csv(
    "~/Downloads/archive/Hotel_Room_attributes.csv", index_col="id"
)
price = pd.read_csv("~/Downloads/archive/hotels_RoomPrice.csv", index_col="id")
```


```python
latest_price = price.drop_duplicates(subset="refid", keep="last")[
    [
        "hotelcode",
        "roomtype",
        "onsiterate",
        "roomamenities",
        "maxoccupancy",
        "mealinclusiontype",
    ]
]
latest_price["ratedescription"] = attributes.loc[latest_price.index]["ratedescription"]
latest_price = latest_price.join(
    details[["hotelname", "city", "country", "starrating"]], on="hotelcode"
)
latest_price = latest_price.rename({"ratedescription": "roomdescription"}, axis=1)
latest_price["mealsincluded"] = ~latest_price["mealinclusiontype"].isnull()
latest_price.pop("hotelcode")
latest_price.pop("mealinclusiontype")
latest_price = latest_price.reset_index(drop=True)
latest_price.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>roomtype</th>
      <th>onsiterate</th>
      <th>roomamenities</th>
      <th>maxoccupancy</th>
      <th>roomdescription</th>
      <th>hotelname</th>
      <th>city</th>
      <th>country</th>
      <th>starrating</th>
      <th>mealsincluded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Vacation Home</td>
      <td>636.09</td>
      <td>Air conditioning: ;Closet: ;Fireplace: ;Free W...</td>
      <td>4</td>
      <td>Shower, Kitchenette, 2 bedrooms, 1 double bed ...</td>
      <td>Pantlleni</td>
      <td>Beddgelert</td>
      <td>United Kingdom</td>
      <td>3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Vacation Home</td>
      <td>591.74</td>
      <td>Air conditioning: ;Closet: ;Dishwasher: ;Firep...</td>
      <td>4</td>
      <td>Shower, Kitchenette, 2 bedrooms, 1 double bed ...</td>
      <td>Willow Cottage</td>
      <td>Beverley</td>
      <td>United Kingdom</td>
      <td>3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Guest room, Queen or Twin/Single Bed(s)</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>AC Hotel Manchester Salford Quays</td>
      <td>Manchester</td>
      <td>United Kingdom</td>
      <td>4</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bargemaster King Accessible Room</td>
      <td>379.08</td>
      <td>Air conditioning: ;Free Wi-Fi in all rooms!: ;...</td>
      <td>2</td>
      <td>Shower</td>
      <td>Lincoln Plaza London, Curio Collection by Hilton</td>
      <td>London</td>
      <td>United Kingdom</td>
      <td>4</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Twin Room</td>
      <td>156.17</td>
      <td>Additional toilet: ;Air conditioning: ;Blackou...</td>
      <td>2</td>
      <td>Room size: 15 m²/161 ft², Non-smoking, Shower,...</td>
      <td>Ibis London Canning Town</td>
      <td>London</td>
      <td>United Kingdom</td>
      <td>3</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



## Describe data attributes

We'll use a self-query retriever, which requires us to describe the metadata we can filter on.

Or if we're feeling lazy we can have a model write a draft of the descriptions for us :)


```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")
res = model.predict(
    "Below is a table with information about hotel rooms. "
    "Return a JSON list with an entry for each column. Each entry should have "
    '{"name": "column name", "description": "column description", "type": "column data type"}'
    f"\n\n{latest_price.head()}\n\nJSON:\n"
)
```


```python
import json

attribute_info = json.loads(res)
attribute_info
```




    [{'name': 'roomtype', 'description': 'The type of the room', 'type': 'string'},
     {'name': 'onsiterate',
      'description': 'The rate of the room',
      'type': 'float'},
     {'name': 'roomamenities',
      'description': 'Amenities available in the room',
      'type': 'string'},
     {'name': 'maxoccupancy',
      'description': 'Maximum number of people that can occupy the room',
      'type': 'integer'},
     {'name': 'roomdescription',
      'description': 'Description of the room',
      'type': 'string'},
     {'name': 'hotelname', 'description': 'Name of the hotel', 'type': 'string'},
     {'name': 'city',
      'description': 'City where the hotel is located',
      'type': 'string'},
     {'name': 'country',
      'description': 'Country where the hotel is located',
      'type': 'string'},
     {'name': 'starrating',
      'description': 'Star rating of the hotel',
      'type': 'integer'},
     {'name': 'mealsincluded',
      'description': 'Whether meals are included or not',
      'type': 'boolean'}]



For low cardinality features, let's include the valid values in the description


```python
latest_price.nunique()[latest_price.nunique() < 40]
```




    maxoccupancy     19
    country          29
    starrating        3
    mealsincluded     2
    dtype: int64




```python
attribute_info[-2]["description"] += (
    f". Valid values are {sorted(latest_price['starrating'].value_counts().index.tolist())}"
)
attribute_info[3]["description"] += (
    f". Valid values are {sorted(latest_price['maxoccupancy'].value_counts().index.tolist())}"
)
attribute_info[-3]["description"] += (
    f". Valid values are {sorted(latest_price['country'].value_counts().index.tolist())}"
)
```


```python
attribute_info
```




    [{'name': 'roomtype', 'description': 'The type of the room', 'type': 'string'},
     {'name': 'onsiterate',
      'description': 'The rate of the room',
      'type': 'float'},
     {'name': 'roomamenities',
      'description': 'Amenities available in the room',
      'type': 'string'},
     {'name': 'maxoccupancy',
      'description': 'Maximum number of people that can occupy the room. Valid values are [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 24]',
      'type': 'integer'},
     {'name': 'roomdescription',
      'description': 'Description of the room',
      'type': 'string'},
     {'name': 'hotelname', 'description': 'Name of the hotel', 'type': 'string'},
     {'name': 'city',
      'description': 'City where the hotel is located',
      'type': 'string'},
     {'name': 'country',
      'description': "Country where the hotel is located. Valid values are ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom']",
      'type': 'string'},
     {'name': 'starrating',
      'description': 'Star rating of the hotel. Valid values are [2, 3, 4]',
      'type': 'integer'},
     {'name': 'mealsincluded',
      'description': 'Whether meals are included or not',
      'type': 'boolean'}]



## Creating a query constructor chain

Let's take a look at the chain that will convert natural language requests into structured queries.

To start we can just load the prompt and see what it looks like


```python
from langchain.chains.query_constructor.base import (
    get_query_constructor_prompt,
    load_query_constructor_runnable,
)
```


```python
doc_contents = "Detailed description of a hotel room"
prompt = get_query_constructor_prompt(doc_contents, attribute_info)
print(prompt.format(query="{query}"))
```

    Your goal is to structure the user's query to match the request schema provided below.
    
    << Structured Request Schema >>
    When responding use a markdown code snippet with a JSON object formatted in the following schema:
    
    ```json
    {
        "query": string \ text string to compare to document contents
        "filter": string \ logical condition statement for filtering documents
    }
    ```
    
    The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.
    
    A logical condition statement is composed of one or more comparison and logical operation statements.
    
    A comparison statement takes the form: `comp(attr, val)`:
    - `comp` (eq | ne | gt | gte | lt | lte | contain | like | in | nin): comparator
    - `attr` (string):  name of attribute to apply the comparison to
    - `val` (string): is the comparison value
    
    A logical operation statement takes the form `op(statement1, statement2, ...)`:
    - `op` (and | or | not): logical operator
    - `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to
    
    Make sure that you only use the comparators and logical operators listed above and no others.
    Make sure that filters only refer to attributes that exist in the data source.
    Make sure that filters only use the attributed names with its function names if there are functions applied on them.
    Make sure that filters only use format `YYYY-MM-DD` when handling timestamp data typed values.
    Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
    Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.
    
    << Example 1. >>
    Data Source:
    ```json
    {
        "content": "Lyrics of a song",
        "attributes": {
            "artist": {
                "type": "string",
                "description": "Name of the song artist"
            },
            "length": {
                "type": "integer",
                "description": "Length of the song in seconds"
            },
            "genre": {
                "type": "string",
                "description": "The song genre, one of "pop", "rock" or "rap""
            }
        }
    }
    ```
    
    User Query:
    What are songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes long in the dance pop genre
    
    Structured Request:
    ```json
    {
        "query": "teenager love",
        "filter": "and(or(eq(\"artist\", \"Taylor Swift\"), eq(\"artist\", \"Katy Perry\")), lt(\"length\", 180), eq(\"genre\", \"pop\"))"
    }
    ```
    
    
    << Example 2. >>
    Data Source:
    ```json
    {
        "content": "Lyrics of a song",
        "attributes": {
            "artist": {
                "type": "string",
                "description": "Name of the song artist"
            },
            "length": {
                "type": "integer",
                "description": "Length of the song in seconds"
            },
            "genre": {
                "type": "string",
                "description": "The song genre, one of "pop", "rock" or "rap""
            }
        }
    }
    ```
    
    User Query:
    What are songs that were not published on Spotify
    
    Structured Request:
    ```json
    {
        "query": "",
        "filter": "NO_FILTER"
    }
    ```
    
    
    << Example 3. >>
    Data Source:
    ```json
    {
        "content": "Detailed description of a hotel room",
        "attributes": {
        "roomtype": {
            "description": "The type of the room",
            "type": "string"
        },
        "onsiterate": {
            "description": "The rate of the room",
            "type": "float"
        },
        "roomamenities": {
            "description": "Amenities available in the room",
            "type": "string"
        },
        "maxoccupancy": {
            "description": "Maximum number of people that can occupy the room. Valid values are [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 24]",
            "type": "integer"
        },
        "roomdescription": {
            "description": "Description of the room",
            "type": "string"
        },
        "hotelname": {
            "description": "Name of the hotel",
            "type": "string"
        },
        "city": {
            "description": "City where the hotel is located",
            "type": "string"
        },
        "country": {
            "description": "Country where the hotel is located. Valid values are ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom']",
            "type": "string"
        },
        "starrating": {
            "description": "Star rating of the hotel. Valid values are [2, 3, 4]",
            "type": "integer"
        },
        "mealsincluded": {
            "description": "Whether meals are included or not",
            "type": "boolean"
        }
    }
    }
    ```
    
    User Query:
    {query}
    
    Structured Request:
    



```python
chain = load_query_constructor_runnable(
    ChatOpenAI(model="gpt-3.5-turbo", temperature=0), doc_contents, attribute_info
)
```


```python
chain.invoke({"query": "I want a hotel in Southern Europe and my budget is 200 bucks."})
```




    StructuredQuery(query='hotel', filter=Operation(operator=<Operator.AND: 'and'>, arguments=[Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Italy'), Comparison(comparator=<Comparator.LTE: 'lte'>, attribute='onsiterate', value=200)]), limit=None)




```python
chain.invoke(
    {
        "query": "Find a 2-person room in Vienna or London, preferably with meals included and AC"
    }
)
```




    StructuredQuery(query='2-person room', filter=Operation(operator=<Operator.AND: 'and'>, arguments=[Operation(operator=<Operator.OR: 'or'>, arguments=[Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='city', value='Vienna'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='city', value='London')]), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='maxoccupancy', value=2), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='mealsincluded', value=True), Comparison(comparator=<Comparator.CONTAIN: 'contain'>, attribute='roomamenities', value='AC')]), limit=None)



## Refining attribute descriptions

We can see that at least two issues above. First is that when we ask for a Southern European destination we're only getting a filter for Italy, and second when we ask for AC we get a literal string lookup for AC (which isn't so bad but will miss things like 'Air conditioning').

As a first step, let's try to update our description of the 'country' attribute to emphasize that equality should only be used when a specific country is mentioned.


```python
attribute_info[-3]["description"] += (
    ". NOTE: Only use the 'eq' operator if a specific country is mentioned. If a region is mentioned, include all relevant countries in filter."
)
chain = load_query_constructor_runnable(
    ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    doc_contents,
    attribute_info,
)
```


```python
chain.invoke({"query": "I want a hotel in Southern Europe and my budget is 200 bucks."})
```




    StructuredQuery(query='hotel', filter=Operation(operator=<Operator.AND: 'and'>, arguments=[Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='mealsincluded', value=False), Comparison(comparator=<Comparator.LTE: 'lte'>, attribute='onsiterate', value=200), Operation(operator=<Operator.OR: 'or'>, arguments=[Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Italy'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Spain'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Greece'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Portugal'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Croatia'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Cyprus'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Malta'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Bulgaria'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Romania'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Slovenia'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Czech Republic'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Slovakia'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Hungary'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Poland'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Estonia'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Latvia'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='country', value='Lithuania')])]), limit=None)



## Refining which attributes to filter on

This seems to have helped! Now let's try to narrow the attributes we're filtering on. More freeform attributes we can leave to the main query, which is better for capturing semantic meaning than searching for specific substrings.


```python
content_attr = ["roomtype", "roomamenities", "roomdescription", "hotelname"]
doc_contents = "A detailed description of a hotel room, including information about the room type and room amenities."
filter_attribute_info = tuple(
    ai for ai in attribute_info if ai["name"] not in content_attr
)
chain = load_query_constructor_runnable(
    ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    doc_contents,
    filter_attribute_info,
)
```


```python
chain.invoke(
    {
        "query": "Find a 2-person room in Vienna or London, preferably with meals included and AC"
    }
)
```




    StructuredQuery(query='2-person room', filter=Operation(operator=<Operator.AND: 'and'>, arguments=[Operation(operator=<Operator.OR: 'or'>, arguments=[Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='city', value='Vienna'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='city', value='London')]), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='maxoccupancy', value=2), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='mealsincluded', value=True)]), limit=None)



## Adding examples specific to our use case

We've removed the strict filter for 'AC' but it's still not being included in the query string. Our chain prompt is a few-shot prompt with some default examples. Let's see if adding use case-specific examples will help:


```python
examples = [
    (
        "I want a hotel in the Balkans with a king sized bed and a hot tub. Budget is $300 a night",
        {
            "query": "king-sized bed, hot tub",
            "filter": 'and(in("country", ["Bulgaria", "Greece", "Croatia", "Serbia"]), lte("onsiterate", 300))',
        },
    ),
    (
        "A room with breakfast included for 3 people, at a Hilton",
        {
            "query": "Hilton",
            "filter": 'and(eq("mealsincluded", true), gte("maxoccupancy", 3))',
        },
    ),
]
prompt = get_query_constructor_prompt(
    doc_contents, filter_attribute_info, examples=examples
)
print(prompt.format(query="{query}"))
```

    Your goal is to structure the user's query to match the request schema provided below.
    
    << Structured Request Schema >>
    When responding use a markdown code snippet with a JSON object formatted in the following schema:
    
    ```json
    {
        "query": string \ text string to compare to document contents
        "filter": string \ logical condition statement for filtering documents
    }
    ```
    
    The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.
    
    A logical condition statement is composed of one or more comparison and logical operation statements.
    
    A comparison statement takes the form: `comp(attr, val)`:
    - `comp` (eq | ne | gt | gte | lt | lte | contain | like | in | nin): comparator
    - `attr` (string):  name of attribute to apply the comparison to
    - `val` (string): is the comparison value
    
    A logical operation statement takes the form `op(statement1, statement2, ...)`:
    - `op` (and | or | not): logical operator
    - `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to
    
    Make sure that you only use the comparators and logical operators listed above and no others.
    Make sure that filters only refer to attributes that exist in the data source.
    Make sure that filters only use the attributed names with its function names if there are functions applied on them.
    Make sure that filters only use format `YYYY-MM-DD` when handling timestamp data typed values.
    Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
    Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.
    
    << Data Source >>
    ```json
    {
        "content": "A detailed description of a hotel room, including information about the room type and room amenities.",
        "attributes": {
        "onsiterate": {
            "description": "The rate of the room",
            "type": "float"
        },
        "maxoccupancy": {
            "description": "Maximum number of people that can occupy the room. Valid values are [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 24]",
            "type": "integer"
        },
        "city": {
            "description": "City where the hotel is located",
            "type": "string"
        },
        "country": {
            "description": "Country where the hotel is located. Valid values are ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom']. NOTE: Only use the 'eq' operator if a specific country is mentioned. If a region is mentioned, include all relevant countries in filter.",
            "type": "string"
        },
        "starrating": {
            "description": "Star rating of the hotel. Valid values are [2, 3, 4]",
            "type": "integer"
        },
        "mealsincluded": {
            "description": "Whether meals are included or not",
            "type": "boolean"
        }
    }
    }
    ```
    
    
    << Example 1. >>
    User Query:
    I want a hotel in the Balkans with a king sized bed and a hot tub. Budget is $300 a night
    
    Structured Request:
    ```json
    {
        "query": "king-sized bed, hot tub",
        "filter": "and(in(\"country\", [\"Bulgaria\", \"Greece\", \"Croatia\", \"Serbia\"]), lte(\"onsiterate\", 300))"
    }
    ```
    
    
    << Example 2. >>
    User Query:
    A room with breakfast included for 3 people, at a Hilton
    
    Structured Request:
    ```json
    {
        "query": "Hilton",
        "filter": "and(eq(\"mealsincluded\", true), gte(\"maxoccupancy\", 3))"
    }
    ```
    
    
    << Example 3. >>
    User Query:
    {query}
    
    Structured Request:
    



```python
chain = load_query_constructor_runnable(
    ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    doc_contents,
    filter_attribute_info,
    examples=examples,
)
```


```python
chain.invoke(
    {
        "query": "Find a 2-person room in Vienna or London, preferably with meals included and AC"
    }
)
```




    StructuredQuery(query='2-person room, meals included, AC', filter=Operation(operator=<Operator.AND: 'and'>, arguments=[Operation(operator=<Operator.OR: 'or'>, arguments=[Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='city', value='Vienna'), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='city', value='London')]), Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='mealsincluded', value=True)]), limit=None)



This seems to have helped! Let's try another complex query:


```python
chain.invoke(
    {
        "query": "I want to stay somewhere highly rated along the coast. I want a room with a patio and a fireplace."
    }
)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    File ~/langchain/libs/langchain/langchain/chains/query_constructor/base.py:53, in StructuredQueryOutputParser.parse(self, text)
         52 else:
    ---> 53     parsed["filter"] = self.ast_parse(parsed["filter"])
         54 if not parsed.get("limit"):


    File ~/langchain/.venv/lib/python3.9/site-packages/lark/lark.py:652, in Lark.parse(self, text, start, on_error)
        635 """Parse the given text, according to the options provided.
        636 
        637 Parameters:
       (...)
        650 
        651 """
    --> 652 return self.parser.parse(text, start=start, on_error=on_error)


    File ~/langchain/.venv/lib/python3.9/site-packages/lark/parser_frontends.py:101, in ParsingFrontend.parse(self, text, start, on_error)
        100 stream = self._make_lexer_thread(text)
    --> 101 return self.parser.parse(stream, chosen_start, **kw)


    File ~/langchain/.venv/lib/python3.9/site-packages/lark/parsers/lalr_parser.py:41, in LALR_Parser.parse(self, lexer, start, on_error)
         40 try:
    ---> 41     return self.parser.parse(lexer, start)
         42 except UnexpectedInput as e:


    File ~/langchain/.venv/lib/python3.9/site-packages/lark/parsers/lalr_parser.py:171, in _Parser.parse(self, lexer, start, value_stack, state_stack, start_interactive)
        170     return InteractiveParser(self, parser_state, parser_state.lexer)
    --> 171 return self.parse_from_state(parser_state)


    File ~/langchain/.venv/lib/python3.9/site-packages/lark/parsers/lalr_parser.py:184, in _Parser.parse_from_state(self, state, last_token)
        183 for token in state.lexer.lex(state):
    --> 184     state.feed_token(token)
        186 end_token = Token.new_borrow_pos('$END', '', token) if token else Token('$END', '', 0, 1, 1)


    File ~/langchain/.venv/lib/python3.9/site-packages/lark/parsers/lalr_parser.py:150, in ParserState.feed_token(self, token, is_end)
        148     s = []
    --> 150 value = callbacks[rule](s)
        152 _action, new_state = states[state_stack[-1]][rule.origin.name]


    File ~/langchain/.venv/lib/python3.9/site-packages/lark/parse_tree_builder.py:153, in ChildFilterLALR_NoPlaceholders.__call__(self, children)
        152         filtered.append(children[i])
    --> 153 return self.node_builder(filtered)


    File ~/langchain/.venv/lib/python3.9/site-packages/lark/parse_tree_builder.py:325, in apply_visit_wrapper.<locals>.f(children)
        323 @wraps(func)
        324 def f(children):
    --> 325     return wrapper(func, name, children, None)


    File ~/langchain/.venv/lib/python3.9/site-packages/lark/visitors.py:501, in _vargs_inline(f, _data, children, _meta)
        500 def _vargs_inline(f, _data, children, _meta):
    --> 501     return f(*children)


    File ~/langchain/.venv/lib/python3.9/site-packages/lark/visitors.py:479, in _VArgsWrapper.__call__(self, *args, **kwargs)
        478 def __call__(self, *args, **kwargs):
    --> 479     return self.base_func(*args, **kwargs)


    File ~/langchain/libs/langchain/langchain/chains/query_constructor/parser.py:79, in QueryTransformer.func_call(self, func_name, args)
         78 if self.allowed_attributes and args[0] not in self.allowed_attributes:
    ---> 79     raise ValueError(
         80         f"Received invalid attributes {args[0]}. Allowed attributes are "
         81         f"{self.allowed_attributes}"
         82     )
         83 return Comparison(comparator=func, attribute=args[0], value=args[1])


    ValueError: Received invalid attributes description. Allowed attributes are ['onsiterate', 'maxoccupancy', 'city', 'country', 'starrating', 'mealsincluded']

    
    During handling of the above exception, another exception occurred:


    OutputParserException                     Traceback (most recent call last)

    Cell In[21], line 1
    ----> 1 chain.invoke({"query": "I want to stay somewhere highly rated along the coast. I want a room with a patio and a fireplace."})


    File ~/langchain/libs/langchain/langchain/schema/runnable/base.py:1113, in RunnableSequence.invoke(self, input, config)
       1111 try:
       1112     for i, step in enumerate(self.steps):
    -> 1113         input = step.invoke(
       1114             input,
       1115             # mark each step as a child run
       1116             patch_config(
       1117                 config, callbacks=run_manager.get_child(f"seq:step:{i+1}")
       1118             ),
       1119         )
       1120 # finish the root run
       1121 except BaseException as e:


    File ~/langchain/libs/langchain/langchain/schema/output_parser.py:173, in BaseOutputParser.invoke(self, input, config)
        169 def invoke(
        170     self, input: Union[str, BaseMessage], config: Optional[RunnableConfig] = None
        171 ) -> T:
        172     if isinstance(input, BaseMessage):
    --> 173         return self._call_with_config(
        174             lambda inner_input: self.parse_result(
        175                 [ChatGeneration(message=inner_input)]
        176             ),
        177             input,
        178             config,
        179             run_type="parser",
        180         )
        181     else:
        182         return self._call_with_config(
        183             lambda inner_input: self.parse_result([Generation(text=inner_input)]),
        184             input,
        185             config,
        186             run_type="parser",
        187         )


    File ~/langchain/libs/langchain/langchain/schema/runnable/base.py:633, in Runnable._call_with_config(self, func, input, config, run_type, **kwargs)
        626 run_manager = callback_manager.on_chain_start(
        627     dumpd(self),
        628     input,
        629     run_type=run_type,
        630     name=config.get("run_name"),
        631 )
        632 try:
    --> 633     output = call_func_with_variable_args(
        634         func, input, run_manager, config, **kwargs
        635     )
        636 except BaseException as e:
        637     run_manager.on_chain_error(e)


    File ~/langchain/libs/langchain/langchain/schema/runnable/config.py:173, in call_func_with_variable_args(func, input, run_manager, config, **kwargs)
        171 if accepts_run_manager(func):
        172     kwargs["run_manager"] = run_manager
    --> 173 return func(input, **kwargs)


    File ~/langchain/libs/langchain/langchain/schema/output_parser.py:174, in BaseOutputParser.invoke.<locals>.<lambda>(inner_input)
        169 def invoke(
        170     self, input: Union[str, BaseMessage], config: Optional[RunnableConfig] = None
        171 ) -> T:
        172     if isinstance(input, BaseMessage):
        173         return self._call_with_config(
    --> 174             lambda inner_input: self.parse_result(
        175                 [ChatGeneration(message=inner_input)]
        176             ),
        177             input,
        178             config,
        179             run_type="parser",
        180         )
        181     else:
        182         return self._call_with_config(
        183             lambda inner_input: self.parse_result([Generation(text=inner_input)]),
        184             input,
        185             config,
        186             run_type="parser",
        187         )


    File ~/langchain/libs/langchain/langchain/schema/output_parser.py:225, in BaseOutputParser.parse_result(self, result, partial)
        212 def parse_result(self, result: List[Generation], *, partial: bool = False) -> T:
        213     """Parse a list of candidate model Generations into a specific format.
        214 
        215     The return value is parsed from only the first Generation in the result, which
       (...)
        223         Structured output.
        224     """
    --> 225     return self.parse(result[0].text)


    File ~/langchain/libs/langchain/langchain/chains/query_constructor/base.py:60, in StructuredQueryOutputParser.parse(self, text)
         56     return StructuredQuery(
         57         **{k: v for k, v in parsed.items() if k in allowed_keys}
         58     )
         59 except Exception as e:
    ---> 60     raise OutputParserException(
         61         f"Parsing text\n{text}\n raised following error:\n{e}"
         62     )


    OutputParserException: Parsing text
    ```json
    {
        "query": "highly rated, coast, patio, fireplace",
        "filter": "and(eq(\"starrating\", 4), contain(\"description\", \"coast\"), contain(\"description\", \"patio\"), contain(\"description\", \"fireplace\"))"
    }
    ```
     raised following error:
    Received invalid attributes description. Allowed attributes are ['onsiterate', 'maxoccupancy', 'city', 'country', 'starrating', 'mealsincluded']


## Automatically ignoring invalid queries

It seems our model get's tripped up on this more complex query and tries to search over an attribute ('description') that doesn't exist. By setting `fix_invalid=True` in our query constructor chain, we can automatically remove any parts of the filter that is invalid (meaning it's using disallowed operations, comparisons or attributes).


```python
chain = load_query_constructor_runnable(
    ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    doc_contents,
    filter_attribute_info,
    examples=examples,
    fix_invalid=True,
)
```


```python
chain.invoke(
    {
        "query": "I want to stay somewhere highly rated along the coast. I want a room with a patio and a fireplace."
    }
)
```




    StructuredQuery(query='highly rated, coast, patio, fireplace', filter=Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='starrating', value=4), limit=None)



## Using with a self-querying retriever

Now that our query construction chain is in a decent place, let's try using it with an actual retriever. For this example we'll use the [ElasticsearchStore](https://python.langchain.com/docs/integrations/vectorstores/elasticsearch).


```python
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
```

## Populating vectorstore

The first time you run this, uncomment the below cell to first index the data.


```python
# docs = []
# for _, room in latest_price.fillna("").iterrows():
#     doc = Document(
#         page_content=json.dumps(room.to_dict(), indent=2),
#         metadata=room.to_dict()
#     )
#     docs.append(doc)
# vecstore = ElasticsearchStore.from_documents(
#     docs,
#     embeddings,
#     es_url="http://localhost:9200",
#     index_name="hotel_rooms",
#     # strategy=ElasticsearchStore.ApproxRetrievalStrategy(
#     #     hybrid=True,
#     # )
# )
```


```python
vecstore = ElasticsearchStore(
    "hotel_rooms",
    embedding=embeddings,
    es_url="http://localhost:9200",
    # strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True) # seems to not be available in community version
)
```


```python
from langchain.retrievers import SelfQueryRetriever

retriever = SelfQueryRetriever(
    query_constructor=chain, vectorstore=vecstore, verbose=True
)
```


```python
results = retriever.invoke(
    "I want to stay somewhere highly rated along the coast. I want a room with a patio and a fireplace."
)
for res in results:
    print(res.page_content)
    print("\n" + "-" * 20 + "\n")
```

    {
      "roomtype": "Three-Bedroom House With Sea View",
      "onsiterate": 341.75,
      "roomamenities": "Additional bathroom: ;Additional toilet: ;Air conditioning: ;Closet: ;Clothes dryer: ;Coffee/tea maker: ;Dishwasher: ;DVD/CD player: ;Fireplace: ;Free Wi-Fi in all rooms!: ;Full kitchen: ;Hair dryer: ;Heating: ;High chair: ;In-room safe box: ;Ironing facilities: ;Kitchenware: ;Linens: ;Microwave: ;Private entrance: ;Refrigerator: ;Seating area: ;Separate dining area: ;Smoke detector: ;Sofa: ;Towels: ;TV [flat screen]: ;Washing machine: ;",
      "maxoccupancy": 6,
      "roomdescription": "Room size: 125 m\u00b2/1345 ft\u00b2, 2 bathrooms, Shower and bathtub, Shared bathroom, Kitchenette, 3 bedrooms, 1 double bed or 2 single beds or 1 double bed",
      "hotelname": "Downings Coastguard Cottages - Type B-E",
      "city": "Downings",
      "country": "Ireland",
      "starrating": 4,
      "mealsincluded": false
    }
    
    --------------------
    
    {
      "roomtype": "Three-Bedroom House With Sea View",
      "onsiterate": 774.05,
      "roomamenities": "Additional bathroom: ;Additional toilet: ;Air conditioning: ;Closet: ;Clothes dryer: ;Coffee/tea maker: ;Dishwasher: ;DVD/CD player: ;Fireplace: ;Free Wi-Fi in all rooms!: ;Full kitchen: ;Hair dryer: ;Heating: ;High chair: ;In-room safe box: ;Ironing facilities: ;Kitchenware: ;Linens: ;Microwave: ;Private entrance: ;Refrigerator: ;Seating area: ;Separate dining area: ;Smoke detector: ;Sofa: ;Towels: ;TV [flat screen]: ;Washing machine: ;",
      "maxoccupancy": 6,
      "roomdescription": "Room size: 125 m\u00b2/1345 ft\u00b2, 2 bathrooms, Shower and bathtub, Shared bathroom, Kitchenette, 3 bedrooms, 1 double bed or 2 single beds or 1 double bed",
      "hotelname": "Downings Coastguard Cottages - Type B-E",
      "city": "Downings",
      "country": "Ireland",
      "starrating": 4,
      "mealsincluded": false
    }
    
    --------------------
    
    {
      "roomtype": "Four-Bedroom Apartment with Sea View",
      "onsiterate": 501.24,
      "roomamenities": "Additional toilet: ;Air conditioning: ;Carpeting: ;Cleaning products: ;Closet: ;Clothes dryer: ;Clothes rack: ;Coffee/tea maker: ;Dishwasher: ;DVD/CD player: ;Fireplace: ;Free Wi-Fi in all rooms!: ;Full kitchen: ;Hair dryer: ;Heating: ;High chair: ;In-room safe box: ;Ironing facilities: ;Kitchenware: ;Linens: ;Microwave: ;Private entrance: ;Refrigerator: ;Seating area: ;Separate dining area: ;Smoke detector: ;Sofa: ;Toiletries: ;Towels: ;TV [flat screen]: ;Wake-up service: ;Washing machine: ;",
      "maxoccupancy": 9,
      "roomdescription": "Room size: 110 m\u00b2/1184 ft\u00b2, Balcony/terrace, Shower and bathtub, Kitchenette, 4 bedrooms, 1 single bed or 1 queen bed or 1 double bed or 2 single beds",
      "hotelname": "1 Elliot Terrace",
      "city": "Plymouth",
      "country": "United Kingdom",
      "starrating": 4,
      "mealsincluded": false
    }
    
    --------------------
    
    {
      "roomtype": "Three-Bedroom Holiday Home with Terrace and Sea View",
      "onsiterate": 295.83,
      "roomamenities": "Air conditioning: ;Dishwasher: ;Free Wi-Fi in all rooms!: ;Full kitchen: ;Heating: ;In-room safe box: ;Kitchenware: ;Private entrance: ;Refrigerator: ;Satellite/cable channels: ;Seating area: ;Separate dining area: ;Sofa: ;Washing machine: ;",
      "maxoccupancy": 1,
      "roomdescription": "Room size: 157 m\u00b2/1690 ft\u00b2, Balcony/terrace, 3 bathrooms, Shower, Kitchenette, 3 bedrooms, 1 queen bed or 1 queen bed or 1 queen bed or 1 sofa bed",
      "hotelname": "Seaside holiday house Artatore (Losinj) - 17102",
      "city": "Mali Losinj",
      "country": "Croatia",
      "starrating": 4,
      "mealsincluded": false
    }
    
    --------------------
    



```python

```
