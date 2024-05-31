以下是您提供的内容的中文翻译：

---

# 人机协作工具验证

这个指南演示了如何将人工验证添加到任何工具中。我们将通过使用`HumanApprovalCallbackHandler`来实现这一点。

假设我们需要对使用`ShellTool`执行的命令进行人工批准。这样做存在明显的风险。让我们看看如何强制在执行此工具之前进行手动的人类批准。

**注意**：我们通常不建议使用`ShellTool`。有很多方法可以误用它，而且它不是大多数用例所必需的。我们在此处使用它仅作为演示目的。

```python
from langchain.callbacks import HumanApprovalCallbackHandler
from langchain.tools import ShellTool
```


```python
tool = ShellTool()
```


```python
print(tool.run("echo Hello World!"))
```

输出：Hello World!


## 添加人工批准

向工具添加默认的`HumanApprovalCallbackHandler`将使得用户必须手动批准每个输入到工具的命令，在实际执行命令之前。

```python
tool = ShellTool(callbacks=[HumanApprovalCallbackHandler()])
```


```python
print(tool.run("ls /usr"))
```

提示：您是否批准以下输入？除'Y'/'Yes'（不区分大小写）以外的任何回答都将被视为拒绝。

输出：ls /usr

批准后执行：

```python
print(tool.run("ls /private"))
```

提示：您是否批准以下输入？除'Y'/'Yes'（不区分大小写）以外的任何回答都将被视为拒绝。

输出：ls /private

拒绝执行：

```python
tool.run("ls /private")
```

```
---------------------------------------------------------------------------

HumanRejectedException                    Traceback (most recent call last)

Cell In[17], line 1
----> 1 print(tool.run("ls /private"))


File ~/langchain/langchain/tools/base.py:257, in BaseTool.run(self, tool_input, verbose, start_color, color, callbacks, **kwargs)
    255 # TODO: maybe also pass through run_manager is _run supports kwargs
    256 new_arg_supported = signature(self._run).parameters.get("run_manager")
--> 257 run_manager = callback_manager.on_tool_start(
    258     {"name": self.name, "description": self.description},
    259     tool_input if isinstance(tool_input, str) else str(tool_input),
    260     color=start_color,
    261     **kwargs,
    262 )
    263 try:
    264     tool_args, tool_kwargs = self._to_args_and_kwargs(parsed_input)


File ~/langchain/langchain/callbacks/manager.py:672, in CallbackManager.on_tool_start(self, serialized, input_str, run_id, parent_run_id, **kwargs)
    669 if run_id is None:
    670     run_id = uuid4()
--> 672 _handle_event(
    673     self.handlers,
    674     "on_tool_start",
    675     "ignore_agent",
    676     serialized,
    677     input_str,
    678     run_id=run_id,
    679     parent_run_id=self.parent_run_id,
    680     **kwargs,
    681 )
    683 return CallbackManagerForToolRun(
    684     run_id, self.handlers, self.inheritable_handlers, self.parent_run_id
    685 )


File ~/langchain/langchain/callbacks/manager.py:157, in _handle_event(handlers, event_name, ignore_condition_name, *args, **kwargs)
    155 except Exception as e:
    156     if handler.raise_error:
--> 157         raise e