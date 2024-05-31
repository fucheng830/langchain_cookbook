# LangChain 使用手册

一些使用 LangChain 构建应用程序的示例代码，重点在于比[主文档](https://python.langchain.com)中包含的示例更实际和更完整的例子。

Notebook | 描述
:- | :-
[LLaMA2_sql_chat.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/LLaMA2_sql_chat.ipynb) | 构建一个聊天应用程序，该应用程序使用开源的 LLM（llama2）与 SQL 数据库交互，特别是在包含名册的 SQLite 数据库上演示。
[Semi_Structured_RAG.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/Semi_Structured_RAG.ipynb) | 对包含半结构化数据（包括文本和表格）的文档执行检索增强生成（RAG），使用 unstructured 进行解析，使用多向量检索器进行存储，并使用 lcel 实现链。
[Semi_structured_and_multi_moda...](https://github.com/langchain-ai/langchain/tree/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb) | 对包含半结构化数据和图像的文档执行检索增强生成（RAG），使用 unstructured 进行解析，使用多向量检索器进行存储和检索，并使用 lcel 实现链。
[Semi_structured_multi_modal_RA...](https://github.com/langchain-ai/langchain/tree/master/cookbook/Semi_structured_multi_modal_RAG_LLaMA2.ipynb) | 对包含半结构化数据和图像的文档执行检索增强生成（RAG），使用多种工具和方法，如 unstructured 进行解析，多向量检索器进行存储，lcel 实现链，并使用开源语言模型如 llama2、llava 和 gpt4all。
[amazon_personalize_how_to.ipynb](https://github.com/langchain-ai/langchain/blob/master/cookbook/amazon_personalize_how_to.ipynb) | 从 Amazon Personalize 获取个性化推荐，并使用自定义代理构建生成 AI 应用程序。
[analyze_document.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/analyze_document.ipynb) | 分析一个长文档。
[autogpt/autogpt.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/autogpt/autogpt.ipynb) | 使用 langchain 的原语（如 llms、prompttemplates、vectorstores、embeddings 和 tools）实现 autogpt 语言模型。
[autogpt/marathon_times.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/autogpt/marathon_times.ipynb) | 实现 autogpt，用于查找马拉松比赛的获胜时间。
[baby_agi.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/baby_agi.ipynb) | 实现 babyagi，一个可以根据给定目标生成并执行任务的 AI 代理，具有替换特定向量存储/模型提供者的灵活性。
[baby_agi_with_agent.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/baby_agi_with_agent.ipynb) | 在 babyagi notebook 中替换执行链，使用具有工具访问权限的代理，以获得更可靠的信息。
[camel_role_playing.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/camel_role_playing.ipynb) | 实现 camel 框架，用于在大规模语言模型中创建自主合作代理，使用角色扮演和初始提示引导聊天代理完成任务。
[causal_program_aided_language_...](https://github.com/langchain-ai/langchain/tree/master/cookbook/causal_program_aided_language_model.ipynb) | 实现因果程序辅助语言（cpal）链，它通过引入因果结构来防止语言模型中的幻觉，特别是在处理具有嵌套依赖关系的复杂叙述和数学问题时。
[code-analysis-deeplake.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/code-analysis-deeplake.ipynb) | 使用 gpt 和 activeloop 的 deep lake 分析自己的代码库。
[custom_agent_with_plugin_retri...](https://github.com/langchain-ai/langchain/tree/master/cookbook/custom_agent_with_plugin_retrieval.ipynb) | 构建一个可以通过检索工具并围绕 openapi 端点创建自然语言包装器来与 AI 插件交互的自定义代理。
[custom_agent_with_plugin_retri...](https://github.com/langchain-ai/langchain/tree/master/cookbook/custom_agent_with_plugin_retrieval_using_plugnplai.ipynb) | 使用 `plugnplai` 目录中的 AI 插件构建具有插件检索功能的自定义代理。
[databricks_sql_db.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/databricks_sql_db.ipynb) | 连接到 databricks 运行时和 databricks sql。
[deeplake_semantic_search_over_...](https://github.com/langchain-ai/langchain/tree/master/cookbook/deeplake_semantic_search_over_chat.ipynb) | 使用 activeloop 的 deep lake 和 gpt4 执行群聊的语义搜索和问答。
[elasticsearch_db_qa.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/elasticsearch_db_qa.ipynb) | 使用自然语言与 elasticsearch 分析数据库交互，并通过 elasticsearch dsl API 构建搜索查询。
[extraction_openai_tools.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/extraction_openai_tools.ipynb) | 使用 OpenAI 工具进行结构化数据提取。
[forward_looking_retrieval_augm...](https://github.com/langchain-ai/langchain/tree/master/cookbook/forward_looking_retrieval_augmented_generation.ipynb) | 实现前瞻性主动检索增强生成（flare）方法，该方法生成问题的答案，识别不确定的词元，基于这些词元生成假设问题，并检索相关文档以继续生成答案。
[generative_agents_interactive_...](https://github.com/langchain-ai/langchain/tree/master/cookbook/generative_agents_interactive_simulacra_of_human_behavior.ipynb) | 实现一个模拟人类行为的生成代理，基于一篇研究论文，使用由 langchain 检索器支持的时间加权记忆对象。
[gymnasium_agent_simulation.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/gymnasium_agent_simulation.ipynb) | 在模拟环境（如文本游戏）中创建简单的代理-环境交互循环，使用 gymnasium。
[hugginggpt.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/hugginggpt.ipynb) | 实现 hugginggpt，一个通过 hugging face 将语言模型（如 chatgpt）与机器学习社区连接的系统。
[hypothetical_document_embeddin...](https://github.com/langchain-ai/langchain/tree/master/cookbook/hypothetical_document_embeddings.ipynb) | 通过假设文档嵌入（hyde）技术生成和嵌入查询的假设答案来改进文档索引。
[learned_prompt_optimization.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/learned_prompt_optimization.ipynb) | 使用强化学习自动增强语言模型提示，可以根据用户偏好定制响应。
[llm_bash.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/llm_bash.ipynb) | 使用语言学习模型（llms）和 bash 进程执行简单的文件系统命令。
[llm_checker.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/llm_checker.ipynb) | 使用 llmcheckerchain 函数创建自检链。
[llm_math.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/llm_math.ipynb) | 使用语言模型和 python repls 解决复杂的文字数学问题。
[llm_summarization_checker.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/llm_summarization_checker.ipynb) | 检查文本摘要的准确性，可以选择多次运行检查以获得更好的结果。
[llm_symbolic_math.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/llm_symbolic_math.ipynb) | 使用语言学习模型（llms）和 sympy（一种用于符号数学的 python 库）求解代数方程。
[meta_prompt.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/meta_prompt.ipynb) | 实现 meta-prompt 概念，这是一种构建自我改进代理的方法，这些代理会反思自己的表现并相应地修改指令。
