# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

default_system_prompt_zh = """# 角色设定
你是一位联网信息搜索专家，你需要根据用户的问题，通过联网搜索来搜集相关信息，然后根据这些信息来回答用户的问题。

# 任务描述
当你接收到用户的问题后，你需要充分理解用户的需求，利用我提供给你的工具，获取相对应的信息、资料，以解答用户的问题。
以下是你在执行任务过程中需要遵循的原则：
- 充分理解用户需求：你需要全面分析和理解用户的问题，必要时对用户的问题进行拆解，以确保领会到用户问题的主要意图。
- 灵活使用工具：当你充分理解用户需求后，请你使用我提供的工具获取信息；当你认为上次工具获取到的信息不全或者有误，以至于不足以回答用户问题时，请思考还需要搜索什么信息，再次调用工具获取信息，直至信息完备。"""

default_system_prompt_en = """# Role
You are an expert in online search. You task is gathering relevant information using advanced online search tools based on the user's query, and providing accurate answers according to the search results.

# Task Description
Upon receiving the user's query, you must thoroughly analyze and understand the user's requirements. In order to effectively address the user's query, you should make the best use of the provided tools to acquire comprehensive and reliable information and data. Below are the principles you should adhere to while performing this task:

- Fully understand the user's needs: Analyze the user's query, if necessary, break it down into smaller components to ensure a clear understanding of the user's primary intent.
- Flexibly use tools: After fully comprehending the user's needs, employ the provided tools to retrieve the necessary information.If the information retrieved previously is deemed incomplete or inaccurate and insufficient to answer the user's query, reassess what additional information is required and invoke the tool again until all necessary data is obtained."""


multi_agent_default_system_prompt_zh = """# 角色设定
你是一位专业、细心的信息收集和整理专家。你能够充分理解用户需求、熟练使用搜索工具，以最高的效率完成用户布置的任务。

# 任务描述
当你接收到用户的问题后，你需要充分理解用户的需求，并思考和规划如何高效快速地完成用户布置的任务。
为了帮助你更好、更快地完成任务，我给你提供了三种工具：
1. 搜索工具：你可以利用搜索引擎进行信息的检索；
2. 网页链接浏览工具：可以打开链接（可以是网页、pdf等）并根据需求描述汇总页面上的所有相关信息。
3. Sub Agent：Sub Agent能够根据你输入的prompt来完成各种类型的任务，Sub Agent自身也可以使用搜索工具或网页链接浏览工具。你可以根据自己的需要，将自己的任务拆分成多个子任务，然后创建一个或多个Agent来帮助你并行完成这些子任务。
"""


multi_agent_default_system_prompt_en = """# Role
You are a professional and meticulous expert in information collection and collation. You can fully understand users' needs, skillfully use search tools, and complete the tasks assigned by users with the highest efficiency.

# Task Description
After receiving users' questions, you need to fully understand their needs and think about and plan how to complete the tasks assigned by users efficiently and quickly.
To help you complete tasks better and faster, I have provided you with three tools:
1. Search tool: You can use the search engine to retrieve information;
2. Link reading tool: link reading tool that can open links (which can be web pages, PDFs, etc.) and summarize all relevant information on the page according to the requirement description.
3. Sub Agent: The Sub Agent can complete various types of tasks according to the prompt you input. The Sub Agent itself can also use the search tool and the link reading tool. You can split your tasks into multiple sub-tasks according to your own needs, and then create one or more Agents to help you complete these sub-tasks in parallel.
"""

tools_api_description_zh_map = {
    "search_bing": {
        "type": "function",
        "function": {
            "name": "search_bing",
            "description": "必应网页搜索 API 可提供安全、无广告且具备位置感知能力的搜索结果，能够从数十亿网页文档中提取相关信息。通过一次 API 调用即可利用必应的能力，对数十亿网页、图片、视频和新闻进行搜索，帮助您的用户从万维网中找到他们所需的内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "互联网搜索的关键词"},
                    "offset": {
                        "type": "integer",
                        "description": "偏移量，从0开始，不超过100",
                        "default": 0,
                    },
                    "count": {
                        "type": "integer",
                        "description": "每页返回条数，最多20条，默认10条",
                        "default": 10,
                    },
                    "mkt": {
                        "type": "string",
                        "description": "Market codes（市场代码），搜索中文内容时使用 zh-CN，搜索英文内容时使用 en-US，默认值 zh-CN",
                        "default": "zh-CN",
                    },
                },
            },
            "required": ["query"],
        },
    },
    "create_sub_agents": {
        "type": "function",
        "function": {
            "name": "create_sub_agents",
            "description": "创建agent函数，可以创建一个或多个Agent，每个agent可以根据输入的prompt完成特定的任务。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_agents": {
                        "type": "array",
                        "description": "创建的agent列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "此agent的prompt",
                                },
                                "index": {
                                    "type": "integer",
                                    "description": "此Agent的编号，int类型，每一个agent的编号需要不同。",
                                },
                            },
                        },
                        "required": ["prompt", "index"],
                    },
                    # "required": ["sub_agents"],
                },
            },
        },
    },
    "text_browser_view": {
        "type": "function",
        "function": {
            "name": "text_browser_view",
            "description": "这是一个链接浏览工具，可以打开链接（可以是网页、pdf等）并根据需求描述汇总页面上的所有相关信息。对所有有价值的链接都可以调用该工具来获取信息，有价值的链接包括但不限于以下几种：1.任务中明确提供的网址，2.搜索结果提供的带有相关摘要的网址，3. 之前调用TextBrowserView返回的内容中包含的且判断可能含有有用信息的网址。请尽量避免自己凭空构造链接。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "目标链接，应该是一个完整的url（以 http 开头）",
                    },
                    "description": {
                        "type": "string",
                        "description": "需求描述文本，详细描述在当前url内想要获取的内容",
                    },
                },
            },
        },
    },
    "search_global": {
        "type": "function",
        "function": {
            "name": "search_global",
            "description": "这是一个联网搜索工具，输入搜索问题，返回网页列表与对应的摘要信息。搜索问题应该简洁清晰，复杂问题应该拆解成多步并一步一步搜索。如果没有搜索到有用的页面，可以调整问题描述（如减少限定词、更换搜索思路）后再次搜索。搜索结果质量和语种有关，对于中文资源可以尝试输入中文问题，非中文的资源可以尝试使用英文或对应语种。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索问题",
                    },
                    "count": {
                        "type": "integer",
                        "description": "每页返回条数，最多200条，默认10条",
                        "default": 10,
                    },
                    "summary_type": {
                        "type": "string",
                        "description": "总结类型，可选值为：short, long。默认为short",
                        "default": "short",
                    },
                    "use_english": {
                        "type": "boolean",
                        "description": "是否使用英文进行搜索，默认为false",
                        "default": False,
                    },
                },
            },
        },
    },
}


tools_api_description_en_map = {
    "search_bing": {
        "type": "function",
        "function": {
            "name": "search_bing",
            "description": "The Bing Web Search API can provide search results that are secure, ad-free, and location-aware, and it can extract relevant information from billions of web documents. With just one API call, you can leverage the power of Bing to search billions of web pages, images, videos, and news, helping your users find what they need on the World Wide Web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "The offset to start the search from. The offset must be a number between 0 and 100.",
                        "default": 0,
                    },
                    "count": {
                        "type": "integer",
                        "description": "The number of results to return. The number must be a number between 1 and 20.",
                        "default": 10,
                    },
                    "mkt": {
                        "type": "string",
                        "description": "The market to search in. The market must be a two-letter country code. Use en-US for English searches and zh-CN for searches in China.",
                        "default": "zh-CN",
                    },
                },
            },
            "required": ["query"],
        },
    },
    "create_sub_agents": {
        "type": "function",
        "function": {
            "name": "create_sub_agents",
            "description": "Creates sub-agents that can perform specific tasks based on the input prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_agents": {
                        "type": "array",
                        "description": "The sub-agents to create. Each sub-agent must have a prompt and an index.",
                        "items": {
                            "type": "object",
                            # "required": ["prompt", "index"],
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The prompt for the sub-agent.",
                                },
                                "index": {
                                    "type": "integer",
                                    "description": "The index of the sub-agent. The index must be an integer and unique for each sub-agent.",
                                },
                            },
                        },
                        "required": ["prompt", "index"],
                    },
                    # "required": ["sub_agents"],
                },
            },
        },
    },
    "text_browser_view": {
        "type": "function",
        "function": {
            "name": "text_browser_view",
            "description": "This is a link reading tool that can open links (which can be web pages, PDFs, etc.) and summarize all relevant information on the page according to the requirement description. This tool can be called to obtain information for all valuable links. Valuable links include but are not limited to the following types: 1. URLs explicitly provided in the task; 2. URLs with relevant summaries provided in search results; 3. URLs contained in the content returned by previous calls to TextBrowserView that are judged to potentially contain useful information. Please try to avoid constructing links out of thin air by yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Target link: should be a complete URL (starting with http)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Requirement description text: a detailed description of the content to be obtained within the current URL",
                    },
                },
            },
        },
    },
    "search_global": {
        "type": "function",
        "function": {
            "name": "search_global",
            "description": "This is a search tool. Enter search queries, and it will return a list of web pages along with their corresponding summary information. Search queries should be concise and clear; complex questions should be broken down into multiple steps and searched step by step. If no useful pages are found, you can adjust the question description (such as reducing qualifiers or changing the search approach) and search again. The quality of search results is related to the language: for Chinese resources, you can try entering Chinese queries; for non-Chinese resources, you can try using English or the corresponding language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "question to be searched.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "The number of results to return. Must be less than 200, and default is 10",
                        "default": 10,
                    },
                    "summary_type": {
                        "type": "string",
                        "description": "Summary type, optional values are: short, long. Default is short",
                        "default": "short",
                    },
                    "use_english": {
                        "type": "boolean",
                        "description": "Whether to use English for search, default is false",
                        "default": False,
                    },
                },
            },
        },
    },
}


def get_system_prompt(language: str) -> str:
    if language == "zh":
        return default_system_prompt_zh
    elif language == "en":
        return default_system_prompt_en
    else:
        raise ValueError(f"Unknown language {language}")


def get_multi_agent_system_prompt(language: str) -> str:
    if language == "zh":
        return multi_agent_default_system_prompt_zh
    elif language == "en":
        return multi_agent_default_system_prompt_en
    else:
        raise ValueError(f"Unknown language {language}")


def get_tools_api_description(language: str, func_list: list[str]) -> list[dict]:
    if language == "zh":
        return [tools_api_description_zh_map[k] for k in func_list]
    elif language == "en":
        return [tools_api_description_en_map[k] for k in func_list]
    else:
        raise ValueError(f"Unknown language {language}")
