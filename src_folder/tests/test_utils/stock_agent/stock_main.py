stock_main = r'''

from typing import Dict, Any, List, Optional, Literal, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
import operator
import json
import os

from tools_code import Investment_Database_CREATE, Investment_Database_READ, Stock_Market_Data_API_READ, Notification_Service_CREATE

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class ReactAgentState(MessagesState):
    remaining_steps: int
    structured_response: Any

class GraphState(MessagesState):
    """ The GraphState represents the state of the LangGraph workflow. """
    input: str
    input_type: Optional[str] = None
    investment_details: Optional[dict] = None
    investment_records: Optional[list] = None
    stock_prices: Optional[dict] = None
    portfolio_value: Optional[float] = None
    portfolio_performance: Optional[str] = None
    alert_details: Optional[dict] = None
    active_alerts: Optional[list] = None
    monitored_stock_price: Optional[float] = None
    alert_triggered: Optional[bool] = None
    notification_message: Optional[str] = None
    output: Optional[str] = None

class InvestmentDetails(BaseModel):
    """Details for adding an investment."""
    symbol: str = Field(description="The stock symbol (e.g., 'AAPL').")
    shares: int = Field(description="The number of shares.")
    price: float = Field(description="The purchase price per share.")

class AlertDetails(BaseModel):
    """Details for setting an alert."""
    symbol: str = Field(description="The stock symbol (e.g., 'GOOG').")
    condition: Literal["above", "below"] = Field(description="The condition for the alert (e.g., 'above', 'below').")
    price: float = Field(description="The target price for the alert.")

class InputClassification(BaseModel):
    """Structured output for classifying user input and extracting details."""
    input_type: Literal["add_investment", "get_portfolio_value", "set_alert", "check_alerts", "unknown"] = Field(
        description="The classified type of input, determining the next step in the workflow."
    )
    investment_details: Optional[InvestmentDetails] = Field(
        default=None, description="Extracted details for adding an investment, if applicable."
    )
    alert_details: Optional[AlertDetails] = Field(
        default=None, description="Extracted details for setting an alert, if applicable."
    )

class InvestmentConfirmation(BaseModel):
    """Structured output for confirming investment addition."""
    confirmation_message: str = Field(description="Confirmation message after successfully recording the investment.")

class PortfolioSummary(BaseModel):
    """Structured output for summarizing portfolio value and performance."""
    portfolio_value: float = Field(description="The calculated total value of the portfolio.")
    portfolio_performance: str = Field(description="A summary of the portfolio's performance.")
    output_message: str = Field(description="The final output message to the user with portfolio value and performance.")

class AlertConfirmation(BaseModel):
    """Structured output for confirming alert setting."""
    confirmation_message: str = Field(description="Confirmation message after successfully setting the alert.")

class AlertCheckResult(BaseModel):
    """Structured output for the result of an alert check."""
    alert_triggered: bool = Field(description="Indicates if an alert condition has been met.")
    notification_message: Optional[str] = Field(default=None, description="The message generated for the notification, if an alert was triggered.")
    output_message: str = Field(description="The final output message indicating if an alert was sent.")

def route_input_action(state: GraphState) -> GraphState:
    """
    Node purpose: Analyzes the user's input or trigger to determine the appropriate action and routes to the corresponding node.
    Implementation reasoning: This node uses an LLM with structured output to classify the user's intent and extract relevant details.
                              Structured output ensures that the downstream nodes receive type-safe and well-defined data.
    """
    structured_llm = llm.with_structured_output(InputClassification)
    user_message = state["messages"][-1].content if state["messages"] else state["input"]
    
    prompt = f"""
    Analyze the following input to determine the user's intent and extract any relevant details.
    
    Input: {user_message}
    
    Possible intents are:
    - 'add_investment': User wants to record a new investment. Extract 'symbol', 'shares', and 'price'.
    - 'get_portfolio_value': User wants to know their portfolio's current value and performance.
    - 'set_alert': User wants to set a new stock price alert. Extract 'symbol', 'condition' (e.g., 'above', 'below'), and 'price'.
    - 'check_alerts': System trigger to check active alerts.
    - 'unknown': If the intent cannot be clearly identified.
    
    Provide the output in the specified JSON format.
    """
    
    result: InputClassification = structured_llm.invoke(prompt)
    
    return {
        "input_type": result.input_type,
        "investment_details": result.investment_details.model_dump() if result.investment_details else None,
        "alert_details": result.alert_details.model_dump() if result.alert_details else None,
        "messages": [AIMessage(content=f"Input classified as: {result.input_type}")]
    }

add_investment_tools = [Investment_Database_CREATE]
def add_investment_action(state: GraphState) -> GraphState:
    """
    Node purpose: Records a new investment in the database based on user input.
    Implementation reasoning: This node uses a React agent to interact with the Investment_Database_CREATE tool
                              to store the investment details. Structured output is used to generate a clear confirmation message.
    """
    class CustomClass(BaseModel):
        confirmation_message: str = Field(description="Confirmation message after successfully recording the investment.")

    agent = create_react_agent(
      model=llm,
      prompt="You are an agent that records new investments. Use the Investment_Database_CREATE tool to store the investment details provided. Once the investment is recorded, generate a confirmation message for the user.",
      tools=add_investment_tools,
      state_schema=ReactAgentState,
      response_format=CustomClass
    )

    investment_details = state["investment_details"]
    if not investment_details:
        return {
            "output": "Error: Investment details not provided.",
            "messages": [AIMessage(content="Error: Investment details not provided for adding investment.")]
        }

    # Assuming Investment_Database_CREATE expects a dictionary with specific keys
    # Adjust this based on the actual tool's expected input format
    tool_input = {
        "database_id": os.environ.get("NOTION_DATABASE_ID"), # Replace with actual database ID if needed
        "properties": {
            "Symbol": {"rich_text": [{"text": {"content": investment_details.get("symbol", "")}}]},
            "Shares": {"number": investment_details.get("shares", 0)},
            "Price": {"number": investment_details.get("price", 0.0)}
        }
    }

    result: CustomClass = agent.invoke({"messages": [HumanMessage(content=f"Record investment: {json.dumps(tool_input)}")]})["structured_response"]
    
    return {
        "output": result.confirmation_message,
        "messages": [AIMessage(content=result.confirmation_message)]
    }

get_portfolio_value_tools = [Investment_Database_READ, Stock_Market_Data_API_READ]
def get_portfolio_value_action(state: GraphState) -> GraphState:
    """
    Node purpose: Retrieves investment records and current stock prices to calculate and summarize portfolio value and performance.
    Implementation reasoning: This node uses a React agent to fetch investment records and stock prices.
                              It then uses an LLM with structured output to aggregate, analyze, and generate a summary.
    """
    class CustomClass(BaseModel):
        portfolio_value: float = Field(description="The calculated total value of the portfolio.")
        portfolio_performance: str = Field(description="A summary of the portfolio's performance.")
        output_message: str = Field(description="The final output message to the user with portfolio value and performance.")

    agent = create_react_agent(
      model=llm,
      prompt="You are an agent that retrieves investment records and current stock prices. Use the Investment_Database_READ tool to get all investment records and the Stock_Market_Data_API_READ tool to get real-time stock prices for all relevant equities. Then, calculate the total portfolio value and summarize its performance. Finally, generate a user-friendly output message.",
      tools=get_portfolio_value_tools,
      state_schema=ReactAgentState,
      response_format=CustomClass
    )

    # Invoke the agent to get investment records and stock prices
    # The agent's prompt should guide it to use both tools and then perform calculations
    result: CustomClass = agent.invoke({"messages": [HumanMessage(content=f"Get my portfolio value and performance.")]})["structured_response"]

    return {
        "portfolio_value": result.portfolio_value,
        "portfolio_performance": result.portfolio_performance,
        "output": result.output_message,
        "messages": [AIMessage(content=result.output_message)]
    }

set_alert_tools = [Investment_Database_CREATE]
def set_alert_action(state: GraphState) -> GraphState:
    """
    Node purpose: Sets up a new alert condition in the database based on user-defined criteria.
    Implementation reasoning: This node uses a React agent to interact with the Investment_Database_CREATE tool
                              to store the alert details. Structured output is used to generate a clear confirmation message.
    """
    class CustomClass(BaseModel):
        confirmation_message: str = Field(description="Confirmation message after successfully setting the alert.")

    agent = create_react_agent(
      model=llm,
      prompt="You are an agent that sets up new stock price alerts. Use the Investment_Database_CREATE tool to store the alert details provided. Once the alert is set, generate a confirmation message for the user.",
      tools=set_alert_tools,
      state_schema=ReactAgentState,
      response_format=CustomClass
    )

    alert_details = state["alert_details"]
    if not alert_details:
        return {
            "output": "Error: Alert details not provided.",
            "messages": [AIMessage(content="Error: Alert details not provided for setting alert.")]
        }

    # Assuming Investment_Database_CREATE expects a dictionary with specific keys for alerts
    # Adjust this based on the actual tool's expected input format
    tool_input = {
        "database_id": os.environ.get("NOTION_DATABASE_ID"), # Replace with actual database ID if needed
        "properties": {
            "Symbol": {"rich_text": [{"text": {"content": alert_details.get("symbol", "")}}]},
            "Condition": {"rich_text": [{"text": {"content": alert_details.get("condition", "")}}]},
            "Price": {"number": alert_details.get("price", 0.0)}
        }
    }

    result: CustomClass = agent.invoke({"messages": [HumanMessage(content=f"Set alert: {json.dumps(tool_input)}")]})["structured_response"]
    
    return {
        "output": result.confirmation_message,
        "messages": [AIMessage(content=result.confirmation_message)]
    }

check_alerts_tools = [Investment_Database_READ, Stock_Market_Data_API_READ, Notification_Service_CREATE]
def check_alerts_action(state: GraphState) -> GraphState:
    """
    Node purpose: Periodically checks all active alerts against real-time stock prices and sends notifications if conditions are met.
    Implementation reasoning: This node uses a React agent to retrieve active alerts, fetch stock prices, and send notifications.
                              It then uses an LLM with structured output to analyze the alert conditions and generate a notification message.
    """
    class CustomClass(BaseModel):
        alert_triggered: bool = Field(description="Indicates if an alert condition has been met.")
        notification_message: Optional[str] = Field(default=None, description="The message generated for the notification, if an alert was triggered.")
        output_message: str = Field(description="The final output message indicating if an alert was sent.")

    agent = create_react_agent(
      model=llm,
      prompt="You are an agent that checks active stock alerts. Use the Investment_Database_READ tool to get all active alerts and the Stock_Market_Data_API_READ tool to get real-time stock prices for the monitored equities. For each alert, check if its condition is met. If an alert is triggered, use the Notification_Service_CREATE tool to send a notification. Finally, summarize the outcome of the alert check.",
      tools=check_alerts_tools,
      state_schema=ReactAgentState,
      response_format=CustomClass
    )

    # Invoke the agent to check alerts and send notifications
    result: CustomClass = agent.invoke({"messages": [HumanMessage(content=f"Check all active alerts.")]})["structured_response"]

    return {
        "alert_triggered": result.alert_triggered,
        "notification_message": result.notification_message,
        "output": result.output_message,
        "messages": [AIMessage(content=result.output_message)]
    }

def route_input_type(state: GraphState) -> str:
    """
    Routing function: Determines the next node based on the classified input type.
    Implementation reasoning: This function acts as a conditional router, directing the workflow
                              to the appropriate specialized node based on the 'input_type' field in the state.
    """
    if state["input_type"] == "add_investment":
        return "add_investment"
    elif state["input_type"] == "get_portfolio_value":
        return "get_portfolio_value"
    elif state["input_type"] == "set_alert":
        return "set_alert"
    elif state["input_type"] == "check_alerts":
        return "check_alerts"
    return "__END__" # Fallback for unknown or unhandled input types

workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("route_input", route_input_action)
workflow.add_node("add_investment", add_investment_action)
workflow.add_node("get_portfolio_value", get_portfolio_value_action)
workflow.add_node("set_alert", set_alert_action)
workflow.add_node("check_alerts", check_alerts_action)

# Add edges
workflow.add_edge(START, "route_input")
workflow.add_conditional_edges(
    "route_input",
    route_input_type,
    {
        "add_investment": "add_investment",
        "get_portfolio_value": "get_portfolio_value",
        "set_alert": "set_alert",
        "check_alerts": "check_alerts",
        "__END__": END
    }
)
workflow.add_edge("add_investment", END)
workflow.add_edge("get_portfolio_value", END)
workflow.add_edge("set_alert", END)
workflow.add_edge("check_alerts", END)

checkpointer = InMemorySaver()
app = workflow.compile(
    checkpointer=checkpointer
)

## Required Keys and Credentials
# OPENAI_API_KEY: OpenAI API key for LLM access.
# USER_ID: User ID for Composio tools.
# NOTION_DATABASE_ID: Notion database ID for Investment_Database_CREATE and Investment_Database_READ tools.
# YOUR_API_KEY: API key for IntrinioRealtimeEquitiesClient in Stock_Market_Data_API_READ tool.
# YOUR_SENDGRID_API_KEY: SendGrid API key for Notification_Service_CREATE tool.
# YOUR_TWILIO_SID: Twilio Account SID for Notification_Service_CREATE tool.
# YOUR_TWILIO_TOKEN: Twilio Auth Token for Notification_Service_CREATE tool.
# YOUR_TELEGRAM_BOT_TOKEN: Telegram Bot Token for Notification_Service_CREATE tool.
# YOUR_ONESIGNAL_APP_ID: OneSignal App ID for Notification_Service_CREATE tool.
# YOUR_ONESIGNAL_API_KEY: OneSignal API Key for Notification_Service_CREATE tool.

'''