from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from final_code.states.AgentBuilderState import AgentBuilderState, AgentInstructions
from final_code.utils.dict_to_reactflow import dict_to_tree_positions
from final_code.llms.model_factory import get_model
from final_code.states.NodesAndEdgesSchemas import JSONSchema
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List


llm = get_model()

ENV_VAR_PROMPT = PromptTemplate.from_template("""
Extract all environment variable names from the following Python script. 
Look for variables accessed via os.environ['VAR_NAME'] or os.getenv('VAR_NAME'), and also parse the comment section at the end of the script which explicitly lists required environment variables.

List each unique environment variable name on a new line.

Example output format:
[API_KEY
DATABASE_URL
DEBUG_MODE
LOG_PATH]"
                                               
<PYTHON_CODE>
{python_code}
</PYTHON_CODE>
""")

class envVariableList(BaseModel):
    env_variables: list = Field(description="A list of environment variable names required to run the python code")

def env_var_node(state: AgentBuilderState):
    python_code: AgentInstructions = state["python_code"]

    var_extraction_llm = llm.with_structured_output(envVariableList)
    var_extracted_output: JSONSchema = var_extraction_llm.invoke([HumanMessage(content=ENV_VAR_PROMPT.format(
        python_code=python_code
    ))])

    return {
        "messages": [AIMessage(content="extracted env variables!")],
        "env_variables": var_extracted_output.env_variables
    } 
