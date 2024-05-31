# Tree of Thought (ToT) example

The Tree of Thought (ToT) is a chain that allows you to query a Large Language Model (LLM) using the Tree of Thought technique. This is based on the paper ["Large Language Model Guided Tree-of-Thought"](https://arxiv.org/pdf/2305.08291.pdf)


```python
from langchain_openai import OpenAI

llm = OpenAI(temperature=1, max_tokens=512, model="gpt-3.5-turbo-instruct")
```

    /Users/harrisonchase/.pyenv/versions/3.9.1/envs/langchain/lib/python3.9/site-packages/deeplake/util/check_latest_version.py:32: UserWarning: A newer version of deeplake (3.6.13) is available. It's recommended that you update to the latest version using `pip install -U deeplake`.
      warnings.warn(



```python
sudoku_puzzle = "3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1"
sudoku_solution = "3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1"
problem_description = f"""
{sudoku_puzzle}

- This is a 4x4 Sudoku puzzle.
- The * represents a cell to be filled.
- The | character separates rows.
- At each step, replace one or more * with digits 1-4.
- There must be no duplicate digits in any row, column or 2x2 subgrid.
- Keep the known digits from previous valid thoughts in place.
- Each thought can be a partial or the final solution.
""".strip()
print(problem_description)
```

    3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1
    
    - This is a 4x4 Sudoku puzzle.
    - The * represents a cell to be filled.
    - The | character separates rows.
    - At each step, replace one or more * with digits 1-4.
    - There must be no duplicate digits in any row, column or 2x2 subgrid.
    - Keep the known digits from previous valid thoughts in place.
    - Each thought can be a partial or the final solution.


## Rules Based Checker

Each thought is evaluated by the thought checker and is given a validity type: valid, invalid or partial. A simple checker can be rule based. For example, in the case of a sudoku puzzle, the checker can check if the puzzle is valid, invalid or partial.

In the following code we implement a simple rule based checker for a specific 4x4 sudoku puzzle.



```python
import re
from typing import Tuple

from langchain_experimental.tot.checker import ToTChecker
from langchain_experimental.tot.thought import ThoughtValidity


class MyChecker(ToTChecker):
    def evaluate(
        self, problem_description: str, thoughts: Tuple[str, ...] = ()
    ) -> ThoughtValidity:
        last_thought = thoughts[-1]
        clean_solution = last_thought.replace(" ", "").replace('"', "")
        regex_solution = clean_solution.replace("*", ".").replace("|", "\\|")
        if sudoku_solution in clean_solution:
            return ThoughtValidity.VALID_FINAL
        elif re.search(regex_solution, sudoku_solution):
            return ThoughtValidity.VALID_INTERMEDIATE
        else:
            return ThoughtValidity.INVALID
```

Just testing the MyChecker class above:


```python
checker = MyChecker()
assert (
    checker.evaluate("", ("3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1",))
    == ThoughtValidity.VALID_INTERMEDIATE
)
assert (
    checker.evaluate("", ("3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1",))
    == ThoughtValidity.VALID_FINAL
)
assert (
    checker.evaluate("", ("3,4,1,2|1,2,3,4|2,1,4,3|4,3,*,1",))
    == ThoughtValidity.VALID_INTERMEDIATE
)
assert (
    checker.evaluate("", ("3,4,1,2|1,2,3,4|2,1,4,3|4,*,3,1",))
    == ThoughtValidity.INVALID
)
```

## Tree of Thought Chain

Initialize and run the ToT chain, with maximum number of interactions `k` set to `30` and the maximum number child thoughts `c` set to `8`.


```python
from langchain_experimental.tot.base import ToTChain

tot_chain = ToTChain(
    llm=llm, checker=MyChecker(), k=30, c=5, verbose=True, verbose_llm=False
)
tot_chain.run(problem_description=problem_description)
```

    
    
    [1m> Entering new ToTChain chain...[0m
    Starting the ToT solve procedure.


    /Users/harrisonchase/workplace/langchain/libs/langchain/langchain/chains/llm.py:275: UserWarning: The predict_and_parse method is deprecated, instead pass an output parser directly to LLMChain.
      warnings.warn(


    [31;1m[1;3mThought: 3*,*,2|1*,3,*|*,1,*,3|4,*,*,1
    [0m[31;1m[1;3mThought: 3*,1,2|1*,3,*|*,1,*,3|4,*,*,1
    [0m[31;1m[1;3mThought: 3*,1,2|1*,3,4|*,1,*,3|4,*,*,1
    [0m[31;1m[1;3mThought: 3*,1,2|1*,3,4|*,1,2,3|4,*,*,1
    [0m[31;1m[1;3mThought: 3*,1,2|1*,3,4|2,1,*,3|4,*,*,1
    [0m

    Type <enum 'ThoughtValidity'> not serializable


    [31;1m[1;3mThought: 3,*,*,2|1,*,3,*|*,1,*,3|4,1,*,*
    [0m[31;1m[1;3mThought: 3,*,*,2|*,3,2,*|*,1,*,3|4,1,*,*
    [0m[31;1m[1;3mThought: 3,2,*,2|1,*,3,*|*,1,*,3|4,1,*,*
    [0m[31;1m[1;3mThought: 3,2,*,2|1,*,3,*|1,1,*,3|4,1,*,*
    [0m[31;1m[1;3mThought: 3,2,*,2|1,1,3,*|1,1,*,3|4,1,*,*
    [0m[33;1m[1;3mThought: 3,*,*,2|1,2,3,*|*,1,*,3|4,*,*,1
    [0m[31;1m[1;3m    Thought: 3,1,4,2|1,2,3,4|2,1,4,3|4,3,2,1
    [0m[32;1m[1;3m    Thought: 3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1
    [0m
    [1m> Finished chain.[0m





    '3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1'




```python

```
