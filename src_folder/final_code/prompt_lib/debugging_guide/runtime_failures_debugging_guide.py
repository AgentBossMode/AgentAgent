key_error_accessing= """
<TSG: KEY_ERROR_ACCESSING_STATE_VARIABLE>
1. SUMMARY: When a node attempts to access a state variable that does not exist in the current `GraphState`, a `KeyError` occurs.


2. WHAT TO LOOK FOR IN LOGS (EXAMPLE):
```
KeyError: 'user_query'
During task with name 'generate_kql_query' and id '8c6913bd-3333-75d6-93a0-d9705fad72cf'
/home/user/app.py:118 in generate_kql_query

    user_query = state["user_query"]
KeyError: 'user_query'
```

3. DEBUGGING STEPS:
    a. **Review `python_code.py`**: Examine the `GraphState` definition and the `add_edge` definitions to understand the flow of data between nodes.
    b. **Inspect the Node and Input and see why the 'KeyError' happens**: 
        1. If it's because the node expects a variable that is supposed to be part of the user's initial input (`state["messages"][-1].content`), use that instead.
        2. If it is because one of the paths does populate the variable but another path does not, then you can add an if check (ex: "if "x" in state ...)
        3. If this is additional information that is necessarily required during the start of the workflow, append it to the bottom
            <EXAMPLE>
<STATE_VARS>
PLEASE POPULATE THE FOLLOWING INPUTS:
<STATE_VAR_1>state_var_1</STATE_VAR_1>
<STATE_VAR_2>state_var_2</STATE_VAR_2>
</STATE_VARS>
            </EXAMPLE>
</TSG: KEY_ERROR_ACCESSING_STATE_VARIABLE>
"""

runtime_failures_debugging_guide = """

<TSG1: KEY_ERROR_ACCESSING_STATE_VARIABLE>
1. SUMMARY: When a node attempts to access a state variable that does not exist in the current `GraphState`, a `KeyError` occurs.


2. WHAT TO LOOK FOR IN LOGS (EXAMPLE):
```
KeyError: 'user_query'
During task with name 'generate_kql_query' and id '8c6913bd-3333-75d6-93a0-d9705fad72cf'
/home/user/app.py:118 in generate_kql_query

    user_query = state["user_query"]
KeyError: 'user_query'
```

3. DEBUGGING STEPS:
    a. **Review `python_code.py`**: Examine the `GraphState` definition and the `add_edge` definitions to understand the flow of data between nodes.
    b. **Inspect the Node and Input**: Determine if the `KeyError` occurs because the node expects a variable that is supposed to be part of the user's initial input (`state["messages"][-1].content`).
    c. **If the variable is derived from user input**: Modify the node's code to extract the required information from `state["messages"][-1].content` and assign it to the missing key.
    ```python
    state["user_query"] = state["messages"][-1].content
    ```
    d. **If the variable is additional input not from `state["messages"]`**: Modify `pytest_code.py` to include the missing variable in the `app.invoke` call.
    **FROM THIS:**
    ```python
    result = app.invoke({{"messages": [
            {{
                "role": "user",
                "content": input_query,
            }}]
            }}, config=thread_config)
    ```
    **TO THIS:**
    ```python
    result = app.invoke({{"messages": [
            {{
                "role": "user",
                "content": input_query,
            }}],
            {{"custom_variable": custom_value_as_per_requirement}}
            }}, config=thread_config)
    ```
    e. Additionally, see if you think this field needs to be customized for each parameterized test, in that case modify the params
</TSG1: KEY_ERROR_ACCESSING_STATE_VARIABLE>


<TSG2: OPENAI_400_BADERROR_ADDITIONAL_PROPERTIES_FALSE>
1. SUMMARY: When ChatOpenAI model is called using structured_output, and the PydanticModel internally contains a 'dict' or 'Dict', then we see this error.
    OpenAI does not support vague definitions for structured outputs.

2. WHAT TO LOOK FOR IN LOGS (EXAMPLE):

openai.BadRequestError: Error code: 400 - {'error': {'message': "Invalid schema for response_format 'TableSchemaOutput': In context=('properties', 'table_schema'), 'additionalProperties' is required to be supplied and to be false.", 'type': 'invalid_request_error', 'param': 'response_format', 'code': None}}
During task with name 'generate_structured_response' and id 'bf37bf46-c93d-29d2-90ed-c819dc181191'
During task with name 'get_table_schema' and id '09f06eaa-fc4f-d42a-f3f7-b00467aa7f88'

class TableSchemaOutput(BaseModel):
            table_schema: Dict[str, Any] = Field(description="The schema of the Kusto table, including column names and data types.")
    
        agent = create_react_agent(
            model=llm,
            prompt="You are an expert in Kusto DB. Your task is to retrieve the schema for the specified Kusto table using the 'get_table_schema' tool. Ensure the output is a dictionary representing the table schema.",
            tools=get_table_schema_tools,
            state_schema=ReactAgentState,
            response_format=TableSchemaOutput
        )
    
        # The agent's input should include the current messages for context and the table name.
>       result: TableSchemaOutput = agent.invoke({
            "messages": state["messages"] + [HumanMessage(content=f"Get schema for table: {state['kusto_table_name']}")]
        })["structured_response"]
3. DEBUGGING STEPS:
    a. The issue is that you are using `Dict` inside a Pydantic model, which is unacceptable for OpenAI's structured output calling.
    b. Analyze the code, where this field is being used, and then change this annotation to something deterministic, primitives like bool, int, str or another Pydantic model with the primitives/list of primitives
    c. Make changes where this field is referenced accordingly.


</TSG2: OPENAI_400_BADERROR_ADDITIONAL_PROPERTIES_FALSE>

<TSG3: GRAPH_RECURSION_LIMIT_ERROR>
1. SUMMARY:
    langgraph.errors.GraphRecursionError: Recursion limit of 25 reached without hitting a stop condition. You can increase the limit by setting the `recursion_limit` config key. For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/GRAPH_RECURSION_LIMIT During task with name 'read_email' and id 'bf125e48-c23e-810b-b541-e896db169865'
    /usr/local/lib/python3.12/site-packages/langgraph/pregel/main.py:2675

2. DEBUGGING STEPS: 
    a. Check the tools_code, there might be pydantic models with attributes marked as 'dict' or 'any' or 'Dict', replace these ambiguous declarations
       Either use a str and modiy description to hint that returns a serialized json
       OR make another pydantic model if you are aware how the value would look like.
</TSG3: GRAPH_RECURSION_LIMIT_ERROR>



"""