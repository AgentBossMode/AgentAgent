from typing import  Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src_folder.final_code.utils.create_react_agent_temp import create_react_agent

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class KustoTableSchema(BaseModel):
    column_name: str = Field(description="The name of the column in the Kusto table.")
    column_type: str = Field(description="The data type of the column (e.g., 'string', 'int', 'datetime').")

class KustoTableSchemaOutput(BaseModel):
    schema_string: str = Field(description="A string representation of the table schema, formatted as a JSON string of a list of KustoTableSchema objects.")

def kusto_get_table_schema(table_name: str) -> str:
    """
    Retrieves the schema of a specified Kusto table.

    This tool is essential for understanding the structure of the data and identifying relevant fields for query generation.
    It allows the agent to understand the table's columns and their data types, which is crucial for constructing accurate KQL queries.

    Args:
        table_name (str): The name of the table for which to retrieve the schema.

    Returns:
        str: A JSON string representation of the table schema.

    Example:
        >>> kusto_get_table_schema("StormEvents")
        '[{"column_name": "StartTime", "column_type": "datetime"}, {"column_name": "EndTime", "column_type": "datetime"}, 
        {"column_name": "EpisodeId", "column_type": "int"}, {"column_name": "EventId", "column_type": "int"},
          {"column_name": "State", "column_type": "string"}, {"column_name": "EventType", "column_type": "string"}]'
    """
    MOCK_TOOL_PROMPT = """
    You are a helpful assistant that generates mock data for tool outputs.
    Given the tool's purpose and expected output, generate a realistic mock response.
    """

    INPUT_PROMPT = """
    Tool Docstring: {description}
    Input: table_name: {table_name}
    Generate a mock output for this tool.
    """
    
    description = kusto_get_table_schema.__doc__
    
    result = llm.with_structured_output(KustoTableSchemaOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(table_name=table_name, description=description))
        ]
    )
    return result.schema_string

class KustoQueryResult(BaseModel):
    row_data: str = Field(description="A string representation of a row in the query result, typically a JSON string of key-value pairs.")

class KustoExecuteQueryResult(BaseModel):
    results_string: str = Field(description="A string representation of the query results, formatted as a JSON string of a list of KustoQueryResult objects.")

def kusto_execute_query(query: str) -> str:
    """
    Executes a Kusto Query Language (KQL) query against the Kusto database and retrieves the results.

    This tool allows the agent to run the generated KQL queries against the Kusto DB and retrieve the results.
    It is crucial for obtaining the actual data based on the user's request.

    Args:
        query (str): The KQL query string to be executed.

    Returns:
        str: A JSON string representation of the query results.

    Example:
        >>> kusto_execute_query("StormEvents | take 5")
        '[{"StartTime": "2007-01-01T00:00:00Z", "EndTime": "2007-01-01T00:06:00Z", "EpisodeId": 1, "EventId": 1, "State": "TEXAS", "EventType": "Thunderstorm Wind"}, {"StartTime": "2007-01-01T00:00:00Z", "EndTime": "2007-01-01T00:06:00Z", "EpisodeId": 2, "EventId": 2, "State": "TEXAS", "EventType": "Thunderstorm Wind"}]'
    """
    MOCK_TOOL_PROMPT = """
    You are a helpful assistant that generates mock data for tool outputs.
    Given the tool's purpose and expected output, generate a realistic mock response.
    """

    INPUT_PROMPT = """
    Tool Docstring: {description}
    Input: query: {query}
    Generate a mock output for this tool.
    """
    
    description = kusto_execute_query.__doc__
    
    result = llm.with_structured_output(KustoExecuteQueryResult).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(query=query, description=description))
        ]
    )
    return result.results_string

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(MessagesState):
    """ The GraphState represents the state of the LangGraph workflow. """
    user_query: str
    table_schema: Optional[str] = None
    kusto_query: Optional[str] = None
    query_results: Optional[str] = None
    final_response: Optional[str] = None

class TableName(BaseModel):
    table_name: str = Field(description="The name of the Kusto table to retrieve the schema for.")

def get_table_schema(state: GraphState) -> GraphState:
    """
    Node purpose: Retrieves the schema of the relevant Kusto table to understand its structure and identify fields.
    Implementation reasoning: Uses an LLM with structured output to extract the table name from the user query,
                              then calls the kusto_get_table_schema tool to fetch the schema.
    """
    structured_llm = llm.with_structured_output(TableName)
    
    # Extract table name from the user query
    table_name_prompt = f"From the following user query, identify the Kusto table name: {state['user_query']}"
    table_name_obj: TableName = structured_llm.invoke(table_name_prompt)
    
    # Call the tool to get the table schema
    schema_str = kusto_get_table_schema(table_name=table_name_obj.table_name)
    
    return {
        "table_schema": schema_str,
        "messages": [AIMessage(content=f"Retrieved table schema for {table_name_obj.table_name}.")]
    }

class KustoQuery(BaseModel):
    query: str = Field(description="The generated Kusto Query Language (KQL) query.")

def generate_kusto_query(state: GraphState) -> GraphState:
    """
    Node purpose: Constructs a Kusto Query Language (KQL) query based on the table schema and the user's request.
    Implementation reasoning: Uses an LLM with structured output to generate a KQL query, ensuring the output
                              adheres to the expected query format.
    """
    structured_llm = llm.with_structured_output(KustoQuery)
    
    prompt = (
        f"Based on the following Kusto table schema:\n{state['table_schema']}\n\n"
        f"And the user's natural language query:\n{state['user_query']}\n\n"
        f"Generate a Kusto Query Language (KQL) query that answers the user's request. "
        f"Ensure the query is syntactically correct and uses the provided schema."
    )
    
    kusto_query_obj: KustoQuery = structured_llm.invoke(prompt)
    
    return {
        "kusto_query": kusto_query_obj.query,
        "messages": [AIMessage(content="Generated Kusto query.")]
    }

def execute_kusto_query(state: GraphState) -> GraphState:
    """
    Node purpose: Executes the generated KQL query against the Kusto DB and retrieves the results.
    Implementation reasoning: Calls the kusto_execute_query tool with the generated KQL query.
    """
    query_results_str = kusto_execute_query(query=state["kusto_query"])
    
    return {
        "query_results": query_results_str,
        "messages": [AIMessage(content="Executed Kusto query and retrieved results.")]
    }

class FinalResponse(BaseModel):
    response: str = Field(description="The human-readable response to the user based on the query results.")

def respond_to_user(state: GraphState) -> GraphState:
    """
    Node purpose: Processes the query results and presents them in a human-readable format to the user.
    Implementation reasoning: Uses an LLM with structured output to summarize the query results into a
                              user-friendly response.
    """
    structured_llm = llm.with_structured_output(FinalResponse)
    
    prompt = (
        f"Based on the following Kusto query results:\n{state['query_results']}\n\n"
        f"And the original user query:\n{state['user_query']}\n\n"
        f"Provide a concise, human-readable response to the user. "
        f"Summarize the key findings from the query results."
    )
    
    final_response_obj: FinalResponse = structured_llm.invoke(prompt)
    
    return {
        "final_response": final_response_obj.response,
        "messages": [AIMessage(content=final_response_obj.response)]
    }

workflow = StateGraph(GraphState)

workflow.add_node("get_table_schema", get_table_schema)
workflow.add_node("generate_kusto_query", generate_kusto_query)
workflow.add_node("execute_kusto_query", execute_kusto_query)
workflow.add_node("respond_to_user", respond_to_user)

workflow.add_edge(START, "get_table_schema")
workflow.add_edge("get_table_schema", "generate_kusto_query")
workflow.add_edge("generate_kusto_query", "execute_kusto_query")
workflow.add_edge("execute_kusto_query", "respond_to_user")
workflow.add_edge("respond_to_user", END)

checkpointer = InMemorySaver()
app = workflow.compile(
    checkpointer=checkpointer
)

def initialize_state(input_message: HumanMessage) -> GraphState:
    return GraphState(user_query=input_message.content, messages=[input_message])

## Required Keys and Credentials
# OPENAI_API_KEY: OpenAI API key for accessing GPT models.
# AAD_TENANT_ID: Azure Active Directory Tenant ID for Kusto authentication.