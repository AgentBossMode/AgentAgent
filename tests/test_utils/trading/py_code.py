py_code = """
from typing import Dict, Any, List, Optional, Literal, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent, ToolNode
from composio import Composio
from composio_langchain import LangchainProvider
import os
import pandas as pd
from textblob import TextBlob
import tweepy
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG
import numpy as np
import cvxpy as cp
from operator import add

# Define the GraphState
class GraphState(MessagesState):
    \"\"\"
    The GraphState represents the state of the LangGraph workflow.
    It extends MessagesState to include domain-specific fields.
    \"\"\"
    market_data: Optional[str] = None  # Changed to str to store content directly
    news_sentiment: Optional[str] = None  # Changed to str to store content directly
    market_analysis: Optional[dict] = None
    trading_opportunity: Optional[dict] = None
    risk_assessment: Optional[dict] = None
    trade_status: Optional[str] = None
    position_details: Optional[str] = None  # Changed to str to store content directly
    strategy_adjustment: Optional[str] = None  # Changed to str to store content directly
    human_review_flag: Optional[bool] = None
    messages: List[Any] = Field(default_factory=list) # Ensure messages are initialized


# Tool Definitions
composio = Composio(provider=LangchainProvider())

# Ensure environment variables are accessed correctly using os.getenv
fetch_stock_data_tool = composio.tools.get(user_id=os.getenv("USER_ID"), tools=["FINAGE_GET_STOCK_LAST_QUOTE"])
fetch_financial_news_tool = composio.tools.get(user_id=os.getenv("USER_ID"), tools=["FINAGE_GET_STOCK_MARKET_NEWS"])

@tool
def analyze_social_sentiment(query: str) -> str:
    \"\"\"Tool for performing sentiment analysis on social media and financial forums.\"\"\"
    # Twitter API credentials (fill your own)
    consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
    consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
    access_token = os.getenv('TWITTER_ACCESS_TOKEN')
    access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

    # Authenticate with Twitter API
    try:
        auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
        api = tweepy.API(auth)

        # Fetch tweets
        tweets = api.search_tweets(q=query, lang="en", count=100)

        sentiment_results = []
        # Analyze sentiment
        for tweet in tweets:
            analysis = TextBlob(tweet.text)
            sentiment_results.append({
                "tweet": tweet.text,
                "polarity": analysis.sentiment.polarity,
                "subjectivity": analysis.sentiment.subjectivity
            })
        return str(sentiment_results)
    except Exception as e:
        return f"Error analyzing social sentiment: {{e}}"

@tool
def execute_buy_order(symbol: str, qty: int) -> str:
    \"\"\"Tool for executing a buy order for a given stock.\"\"\"
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')

    try:
        trading_client = TradingClient(api_key, api_secret, paper=True)

        # Buy order
        buy_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        buy_order = trading_client.submit_order(order_data=buy_order_data)
        return f"Buy order submitted: {{buy_order}}"
    except Exception as e:
        return f"Error executing buy order: {{e}}"
    messages = state["messages"]

    prompt = f"Analyze the following market data: {{market_data}}. And news sentiment: {{news_sentiment}}. Identify trends, anomalies, and their potential impact. Also consider the conversation history: {{messages}}"
    
    try:
        result = structured_llm.invoke(prompt)
        # Append the new system message to the existing messages
        updated_messages = messages + [SystemMessage(content=f"Market analysis completed: {{result.market_analysis}}")]
        return {
            "market_analysis": result.market_analysis,
            "messages": updated_messages
        }
    except Exception as e:
        print(f"Error in market_data_analysis: {{e}}")
        return {"messages": messages + [AIMessage(content=f"Error during market data analysis: {e}")]}

class PredictiveModelingOutput(BaseModel):
    trading_opportunity: dict = Field(description="Identified trading opportunities with details like asset, direction, and entry/exit points.")
    risk_assessment: dict = Field(description="Assessment of potential risks associated with the identified trading opportunities, including volatility and downside potential.")

def predictive_modeling(state: GraphState) -> GraphState:
    \"\"\"
    Node purpose: Applies predictive models to forecast price movements, identify trading opportunities, and assess potential risks.
    Implementation reasoning: Uses an LLM with structured output to generate consistent trading opportunities and risk assessments.
    \"\"\"
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    structured_llm = llm.with_structured_output(PredictiveModelingOutput)

    market_analysis = state.get("market_analysis", "No market analysis available.")
    messages = state["messages"]

    prompt = f"Based on the market analysis: {market_analysis}, forecast price movements, identify trading opportunities, and assess potential risks. Consider the conversation history: {messages}"
    
    try:
        result = structured_llm.invoke(prompt)
        # Append the new system message to the existing messages
        updated_messages = messages + [SystemMessage(content=f"Predictive modeling completed. Opportunity: {result.trading_opportunity}, Risk: {result.risk_assessment}")]
        return {
            "trading_opportunity": result.trading_opportunity,
            "risk_assessment": result.risk_assessment,
            "messages": updated_messages
        }
    except Exception as e:
        print(f"Error in predictive_modeling: {e}")
        return {"messages": messages + [AIMessage(content=f"Error during predictive modeling: {e}")]}

trade_execution_tools = [execute_buy_order, execute_sell_order, get_portfolio_positions]
def trade_execution(state: GraphState) -> GraphState:
    \"\"\"
    Node purpose: Executes buy/sell orders and manages positions based on identified trading opportunities and pre-approved strategies. This node operates without direct human validation for pre-approved strategies.
    Implementation reasoning: Uses a tool-calling react agent to execute trades and manage positions.
    \"\"\"
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    agent = create_react_agent(
        model=llm,
        tools=trade_execution_tools,
        prompt="You are an AI agent responsible for executing trades. Based on the identified trading opportunity and risk assessment, execute buy/sell orders and manage positions. Use the provided tools. Current conversation history: {messages}. Trading opportunity: {trading_opportunity}. Risk assessment: {risk_assessment}"
    )
    try:
        # The agent's input should incorporate the conversation history and relevant state variables
        response = agent.invoke({
            "messages": state["messages"],
            "trading_opportunity": state.get("trading_opportunity"),
            "risk_assessment": state.get("risk_assessment")
        })
        
        last_message_content = response["messages"][-1].content if response["messages"] else ""

        # Update messages with the agent's response
        updated_messages = state["messages"] + [response["messages"][-1]] if response["messages"] else state["messages"]

        return {
            "trade_status": "executed", # Placeholder, actual status from tool output
            "position_details": last_message_content, # Assuming the agent returns position details
            "messages": updated_messages
        }
    except Exception as e:
        print(f"Error in trade_execution: {e}")
        return {"messages": state["messages"] + [AIMessage(content=f"Error during trade execution: {e}")]}

risk_management_and_strategy_adjustment_tools = [get_portfolio_positions, update_strategy_parameters, adjust_risk_exposure]
def risk_management_and_strategy_adjustment(state: GraphState) -> GraphState:
    \"\"\"
    Node purpose: Dynamically adjusts trading strategies and risk parameters based on evolving market conditions, volatility, or pre-set thresholds. Also flags significant events for human review.
    Implementation reasoning: Uses a tool-calling react agent to adjust strategies and risk, and an LLM to determine if human review is needed.
    \"\"\"
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    agent = create_react_agent(
        model=llm,
        tools=risk_management_and_strategy_adjustment_tools,
        prompt="You are an AI agent responsible for risk management and strategy adjustment. Based on market conditions, volatility, and pre-set thresholds, adjust trading strategies and risk parameters. Flag significant events for human review. Use the provided tools. Current conversation history: {messages}. Market analysis: {market_analysis}. Risk assessment: {risk_assessment}"
    )
    try:
        # The agent's input should incorporate the conversation history and relevant state variables
        response = agent.invoke({
            "messages": state["messages"],
            "market_analysis": state.get("market_analysis"),
            "risk_assessment": state.get("risk_assessment")
        })

        last_message_content = response["messages"][-1].content if response["messages"] else ""

        # LLM call to determine if human review is needed
        class HumanReviewFlag(BaseModel):
            flag: bool = Field(description="True if human review is needed, False otherwise.")
            reason: Optional[str] = Field(description="Reason for human review, if flagged.")

        structured_llm = llm.with_structured_output(HumanReviewFlag)
        review_prompt = f"Based on the current market conditions and strategy adjustments: {last_message_content}, determine if human review is needed. Provide a reason if flagged. Consider the full conversation history: {state['messages']}"
        review_result = structured_llm.invoke(review_prompt)

        # Update messages with the agent's response and the review result
        updated_messages = state["messages"] + [response["messages"][-1]] if response["messages"] else state["messages"]
        updated_messages.append(SystemMessage(content=f"Human review needed: {review_result.flag}. Reason: {review_result.reason}"))

        return {
            "strategy_adjustment": last_message_content, # Assuming the agent returns adjustment details
            "human_review_flag": review_result.flag,
            "messages": updated_messages
        }
    except Exception as e:
        print(f"Error in risk_management_and_strategy_adjustment: {e}")
        return {"messages": state["messages"] + [AIMessage(content=f"Error during risk management and strategy adjustment: {e}")]}
# Define the graph workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("market_data_ingestion", ingest_market_data)
workflow.add_node("market_data_analysis", market_data_analysis)
workflow.add_node("predictive_modeling", predictive_modeling)
workflow.add_node("trade_execution", trade_execution)
workflow.add_node("risk_management_and_strategy_adjustment", risk_management_and_strategy_adjustment)

# Add edges
workflow.add_edge(START, "market_data_ingestion")
workflow.add_edge("market_data_ingestion", "market_data_analysis")
workflow.add_edge("market_data_analysis", "predictive_modeling")

def route_from_predictive_modeling(state: GraphState) -> Literal["trade_execution", "risk_management_and_strategy_adjustment"]:
    \"\"\"
    Routing function from predictive_modeling.
    If a trading opportunity is identified, proceeds to trade_execution.
    Otherwise, or if human review is needed, proceeds to risk_management_and_strategy_adjustment.
    \"\"\"
    # Check if trading_opportunity exists and is not empty
    if state.get("trading_opportunity") and state["trading_opportunity"]:
        # Assuming 'approved' is a key within trading_opportunity for conditional routing
        # If 'approved' is not explicitly set, default to True to proceed with trade execution
        if state["trading_opportunity"].get("approved", True):
            return "trade_execution"
    
    # If no trading opportunity or not approved, or if human review is flagged, go to risk management
    # The original code had a partial condition, completing it here.
    # It's important to define what constitutes "human review needed" from the state.
    # For now, we'll assume if no trade is executed, it goes to risk management.
    return "risk_management_and_strategy_adjustment"

# Add conditional edge from predictive_modeling
workflow.add_conditional_edges(
    "predictive_modeling",
    route_from_predictive_modeling,
    {
        "trade_execution": "trade_execution",
        "risk_management_and_strategy_adjustment": "risk_management_and_strategy_adjustment",
    },
)

# Add edges for the rest of the workflow
workflow.add_edge("trade_execution", "risk_management_and_strategy_adjustment")

# Define the end point based on the human_review_flag
def route_from_risk_management(state: GraphState) -> Literal["human_review", "END"]:
    \"\"\"
    Routing function from risk_management_and_strategy_adjustment.
    If human review is flagged, proceeds to a hypothetical "human_review" node (or END if not implemented).
    Otherwise, the workflow ends.
    \"\"\"
    if state.get("human_review_flag"):
        # If a human review node exists, route to it. Otherwise, it might be an END or a loop back.
        # For this example, we'll assume "human_review" is a conceptual state that leads to END.
        return "human_review" # This would ideally be a node for human interaction
    return "END"

# Add a conceptual "human_review" node if needed, or directly to END
# For simplicity, we'll route to END after human review or if no review is needed.
workflow.add_conditional_edges(
    "risk_management_and_strategy_adjustment",
    route_from_risk_management,
    {
        "human_review": END, # Assuming human review leads to termination or re-evaluation outside this graph
        "END": END,
    },
)

# Compile the graph
app = workflow.compile()
Show less
"""