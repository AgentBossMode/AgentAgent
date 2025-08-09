from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.final_code.states.AgentBuilderState import AgentBuilderState, AgentInstructions
from src.final_code.llms.model_factory import get_model
from langchain_tavily import TavilySearch, TavilyExtract
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
## Prompt: Extracting Required API Keys from Python Code
You are tasked with analyzing Python code to identify and extract all **API keys** required for the code to execute properly.
---
### Instructions
1. **Identify Required API Keys**
   - Extract all API key variable names used in the code.
   - Include keys for:
     - LLMs such as `openai`, `gemini`, `anthropic`, etc. If the key name for a LLM is not present in any comment, the code or any config block. Use TavilySearch to Find API key name required (continue to step 3)
     - Any **third-party tools that are not part of Composio**
2. **Classify Tools**
   - For **non-Composio tools**, continue to step 3.
3. **Use TavilySearch to Find SDK or Code Samples**
   - For each non-Composio tool, search using:
     ```
     TavilySearch("tool_name Python SDK or usage example")
     ```
4. **Use TavilyExtract to Extract Code**
   - Use TavilyExtract on the search results to extract Python code snippets:
   - Extract code that reveals the API key name, such as environment variable names or parameter names like `api_key=`.
5. **Final Output Format**
   - Return the list of key names (strings) in the following format:
     ```python
     ["OPENAI_API_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY", "PINECONE_API_KEY"]
     ```
---
### Additional Notes
- Do **not** include API keys used for Composio tools.
- Extract only the exact names required to authenticate with each service (e.g., from `os.getenv`, direct variable names, or config parameters).
- Be case-sensitive and preserve the exact naming conventions used in the code or SDK documentation.<PYTHON_CODE>
<PYTHON_CODE>
{python_code}
</PYTHON_CODE>

Now produce the final list of names environment variables
""")
tools=[tavily_search_tool, tavily_extract_tool]

class EnvVariableList(BaseModel):
    env_variables: List[str] = Field(description="List of environment variable names required to run the python code")

def env_var_node(state: AgentBuilderState):
    python_code = state["python_code"]
    env_var_react_agent = create_react_agent(llm, tools=tools, name="get_env_var")
    final_response = env_var_react_agent.invoke(
        {"messages": [HumanMessage(content=ENV_VAR_PROMPT.format(python_code=python_code))]})
    prompt = "You will be provided a potentially containing a list of name of variables, from the given text you nee to extract the list the list of name of vriables and return them"
    var_extraction_llm = llm.with_structured_output(EnvVariableList)
    var_extracted_output = var_extraction_llm.invoke([SystemMessage(content=prompt)] + [HumanMessage(content=final_response["messages"][-1].content)])
    return {
        "messages": [AIMessage(content="extracted env variables!")],
        "env_variables": var_extracted_output.env_variables
    }