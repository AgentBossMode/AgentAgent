from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from final_code.states.AgentBuilderState import AgentBuilderState, AgentInstructions
from final_code.llms.model_factory import get_model
from langchain_tavily import TavilySearch, TavilyExtract
from final_code.states.NodesAndEdgesSchemas import JSONSchema, Tool
from pydantic import BaseModel, Field
from typing import List 


llm = get_model()

tavily_extract_tool = TavilyExtract(
    extract_depth="advanced",
    include_images=False)

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
)

ENV_VAR_PROMPT = PromptTemplate.from_template("""
### Prompt: Extracting API Key Names from Tool Names and Code Snippets

You are an expert in analyzing tool integrations in Python applications. Your task is to identify the **API key variable names** required for each tool listed.

---

### Input Format

You are given:
- A list of tools.
- For each tool, a **Python code snippet** (optional, may be empty).

---

### Instructions

#### 1. Analyze Code Snippet (If Provided)

- Inspect the given code snippet for each tool.
- If it includes API key usage (e.g., `os.getenv("API_KEY")`, `api_key="..."`, `Tool(api_key=...)`), extract the **API key variable name** (e.g., `"OPENAI_API_KEY"`).

#### 2. Fallback to Tavily Search (If Needed)

- If the code snippet does **not** show any API key usage:
  - Run:
    ```python
    TavilySearch("tool_name Python SDK or usage example")
    ```
  - Then use `TavilyExtract` on the result to identify how the API key is set or passed.
  - Extract the **API key name** (e.g., `"TOOLNAME_API_KEY"` or whatever is used in examples).

---

### Final Output Format

Return only a **Python list of strings**, where each string is the API key name:

```python
["OPENAI_API_KEY", "PINECONE_API_KEY", "SERPAPI_API_KEY"]
```

---

### Rules

- Return only the **API key variable name** required to authenticate with each tool.
- Ensure you go through all the tools provided in the input list
- Be **case-sensitive** and **preserve exact naming** from the code snippet or SDK documentation.
- If multiple valid names exist, choose the **most commonly used one** in official sources.

---

### Example Input

```python
tools = [
  {
    "tool": "OpenAI",
    "code": "openai.ChatCompletion.create(model='gpt-4', messages=[...])"
  },
  {
    "tool": "Pinecone",
    "code": "pinecone.init(api_key=os.getenv('PINECONE_API_KEY'))"
  },
  {
    "tool": "SerpAPI",
    "code": ""
  }
]
```

### Example Output

```python
["OPENAI_API_KEY", "PINECONE_API_KEY", "SERPAPI_API_KEY"]
```
<INPUT_LIST>
{tool_list}
</INPUT_LIST>
""")


class envVariableList(BaseModel):
    env_variables: List[str] = Field(description="List of environment variable names required to run the python code")

def env_var_node(state: AgentBuilderState):
    python_code = state["python_code"]
    json_schema: JSONSchema = state["json_schema"]
    tool_dict = []
    for tool in json_schema.tools:
        if not tool.is_composio_tool:
            tool_dict.append({"tool":tool.name,"code":tool.py_code})
    list_str = ''.join([str(d) for d in tool_dict])
    llm_with_tools = llm.bind_tools(tools=[tavily_search_tool, tavily_extract_tool])
    var_extraction_llm = llm_with_tools.with_structured_output(envVariableList)
    var_extracted_output = var_extraction_llm.invoke([HumanMessage(content=ENV_VAR_PROMPT.format(
        tool_list = list_str
    ))])
    print(var_extracted_output)

    return {
        "messages": [AIMessage(content="extracted env variables!")],
        "env_variables": var_extracted_output.env_variables
    } 
