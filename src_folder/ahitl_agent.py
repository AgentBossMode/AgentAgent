"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.
"""

from typing import Any, List
from typing_extensions import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from copilotkit.langgraph import CopilotKitState

class AgentState(CopilotKitState):
    pass
@tool
def ask_question(question: str, uiSchema: str) -> str:
    """Asks the user for structured information by generating a form.

    This tool should be used when you need to gather pieces of information
    from the user to proceed. The `question` parameter must be a serialized JSON string
    that conforms to the react-jsonschema-form (RJSF) schema.

    The agent is responsible for generating both:
      - `question`: A valid RJSF schema describing the data structure.
      - `uiSchema`: A valid RJSF UI schema describing presentation details (widgets, placeholders, etc.).

    Args:
        question: A serialized JSON string representing a form schema adhering to react-jsonschema-form.
        uiSchema: A serialized JSON string representing the UI schema for the form.

    Example Scenarios:
    ------------------

    1. **Scheduling a Meeting**
       User says: "I need to schedule a meeting with the marketing team about the Q3 launch."

       **question:**
       ```json
       {
         "title": "Schedule Meeting",
         "type": "object",
         "properties": {
           "meetingTitle": { "type": "string", "title": "Meeting Title" },
           "attendees": {
             "type": "array",
             "title": "Attendees",
             "items": { "type": "string", "format": "email" }
           },
           "datetime": { "type": "string", "format": "date-time", "title": "Date & Time" },
           "agenda": { "type": "string", "title": "Agenda" }
         },
         "required": ["meetingTitle", "attendees", "datetime"]
       }
       ```

       **uiSchema:**
       ```json
       {
         "meetingTitle": { "ui:placeholder": "Enter meeting title" },
         "attendees": {
           "items": { "ui:placeholder": "email@example.com" }
         },
         "agenda": {
           "ui:widget": "textarea",
           "ui:options": { "rows": 5 }
         },
         "ui:submitButtonOptions": {
           "submitText": "Schedule",
           "props": { "className": "btn-primary" }
         }
       }
       ```

    2. **Searching for a Used Car**
       User says: "I'm looking for a used car, maybe a Honda or Toyota, under $15,000."

       **question:**
       ```json
       {
         "title": "Used Car Search",
         "type": "object",
         "properties": {
           "make": {
             "type": "array",
             "title": "Preferred Makes",
             "items": {
               "type": "string",
               "enum": ["Honda", "Toyota", "Ford", "Hyundai"]
             }
           },
           "model": { "type": "string", "title": "Model" },
           "maxPrice": {
             "type": "number",
             "title": "Maximum Price",
             "maximum": 15000
           }
         }
       }
       ```

       **uiSchema:**
       ```json
       {
         "make": { "ui:widget": "checkboxes" },
         "model": { "ui:placeholder": "Enter model name" },
         "maxPrice": {
           "ui:options": { "inputType": "number" },
           "ui:placeholder": "e.g. 15000"
         },
         "ui:submitButtonOptions": { "submitText": "Search" }
       }
       ```

    3. **Changing Notification Settings**
       User says: "I'm getting too many alerts, I want to change my notification settings."

       **question:**
       ```json
       {
         "title": "Notification Settings",
         "type": "object",
         "properties": {
           "emailAlerts": { "type": "boolean", "title": "Email Alerts" },
           "pushNotifications": { "type": "boolean", "title": "Push Notifications" },
           "dailyDigestTime": {
             "type": "string",
             "format": "time",
             "title": "Daily Digest Time"
           }
         }
       }
       ```

       **uiSchema:**
       ```json
       {
         "emailAlerts": { "ui:widget": "checkbox" },
         "pushNotifications": { "ui:widget": "checkbox" },
         "ui:submitButtonOptions": { "submitText": "Save Settings" }
       }
       ```

    4. **Booking a Flight**
       User says: "Book a flight for me next week."

       **question:**
       ```json
       {
         "title": "Flight Booking Details",
         "type": "object",
         "properties": {
           "departureCity": { "type": "string", "title": "Departure City" },
           "destinationCity": { "type": "string", "title": "Destination City" },
           "departureDate": { "type": "string", "format": "date", "title": "Departure Date" },
           "returnDate": { "type": "string", "format": "date", "title": "Return Date" },
           "preferredAirline": { "type": "string", "title": "Preferred Airline" }
         },
         "required": ["departureCity", "destinationCity", "departureDate"]
       }
       ```

    5. **Confirming Reference (Ambiguous Context)**
       User says: "How much does it cost?"

       **question:**
       ```json
       {
         "title": "Confirm Item",
         "type": "object",
         "properties": {
           "item": {
             "type": "string",
             "title": "Select what you're referring to",
             "enum": ["Laptop", "Monitor", "Mouse"]
           }
         },
         "required": ["item"]
       }
       ```

    ------------------

    ### Common uiSchema Patterns Reference:
    - `"ui:widget": "textarea"` → multi-line text input
    - `"ui:widget": "password"` → password input
    - `"ui:widget": "checkbox"` → boolean toggle
    - `"ui:widget": "checkboxes"` → multi-select array
    - `"ui:placeholder": "..."` → hint text for input fields
    - `"ui:options": {"rows": 5}` → extra widget options
    - `"ui:submitButtonOptions"` → controls submit button label and props
    """

    print(f"Interrupting to ask question: {question}")
    print(f"uiSchema: {uiSchema}")


    # wrap the question into the ahitl_form format
    question = {
        "type": "ahitl_form",
        "title": "Additional Information Required",
        "description": "Please fill out the following form to provide the necessary information.",
        "schema": question,
        "uiSchema": uiSchema
    }
    value = interrupt(question)

    print(f"Received answer: {value}")
    return value

backend_tools = [
    ask_question
]

# Extract tool names from backend_tools for comparison
backend_tool_names = [tool.name for tool in backend_tools]


async def chat_node(state: AgentState, config: RunnableConfig) -> Command[Literal["tool_node", "__end__"]]:
    """
    Standard chat node based on the ReAct design pattern. It handles:
    - The model to use (and binds in CopilotKit actions and the tools defined above)
    - The system prompt
    - Getting a response from the model
    - Handling tool calls

    For more about the ReAct design pattern, see:
    https://www.perplexity.ai/search/react-agents-NcXLQhreS0WDzpVaS4m9Cg
    """

    # 1. Define the model
    model = ChatOpenAI(model="gpt-4o-mini")

    # 2. Bind the tools to the model
    model_with_tools = model.bind_tools(
        [
            *state.get("copilotkit", {}).get("actions", []) ,
            *backend_tools,
            # your_tool_here
        ],

        # 2.1 Disable parallel tool calls to avoid race conditions,
        #     enable this for faster performance if you want to manage
        #     the complexity of running tool calls in parallel.
        parallel_tool_calls=False,
    )

    # 3. Define the system message by which the chat model will be run
    system_message = SystemMessage(
        content="""
        You are a helpful assistant, you are very careful to not take any assumptions.
        You have two tools: ask_question and generateForm tools.
        If the user query says : "build a form...", you must use the generateForm tool.
        For other user queries, If you need more information ALWAYS use the ask_question tool, read the definition of ask_question to see if it fits
        """

    )

    # 4. Run the model to generate a response
    response = await model_with_tools.ainvoke([
        system_message,
        *state["messages"],
    ], config)

    # only route to tool node if tool is not in the tools list
    if route_to_tool_node(response):
        print("routing to tool node")
        return Command(
            goto="tool_node",
            update={
                "messages": [response],
            }
        )

    # 5. We've handled all tool calls, so we can end the graph.
    return Command(
        goto=END,
        update={
            "messages": [response],
        }
    )

def route_to_tool_node(response: BaseMessage):
    """
    Route to tool node if any tool call in the response matches a backend tool name.
    """
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        return False

    for tool_call in tool_calls:
        if tool_call.get("name") in backend_tool_names:
            return True
    return False

# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tool_node", ToolNode(tools=backend_tools))
workflow.add_edge("tool_node", "chat_node")
workflow.set_entry_point("chat_node")

app = workflow.compile()
