nutrition_agent_code = r'''
```python
from typing import Dict, Any, List, Optional, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
import re
import json

from tools_code import CreateCalorieData, ReadCalorieData, SearchFoodCalorieAPI, SearchExerciseCalorieAPI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(MessagesState):
    """
    The GraphState represents the state of the LangGraph workflow.
    It inherits from MessagesState, which provides a 'messages' field for conversation history.
    """
    input: Optional[str] = Field(description="The user's initial input query.")
    action: Optional[str] = Field(description="Determined action based on user input (log, retrieve, calculate).")
    food_item: Optional[str] = Field(description="Name of the food item to log.")
    exercise_activity: Optional[str] = Field(description="Name of the exercise activity to log.")
    calories_consumed: Optional[int] = Field(description="Calories consumed from food.")
    calories_burned: Optional[int] = Field(description="Calories burned from exercise.")
    query_type: Optional[str] = Field(description="Type of data retrieval query (historical, net_calories).")
    historical_data: Optional[dict] = Field(description="Retrieved historical data.")
    net_calories: Optional[int] = Field(description="Calculated net calorie balance.")

class ActionClassifier(BaseModel):
    """Structured output for classifying user intent."""
    action: Literal["log_data", "retrieve_historical_data", "calculate_net_calories"] = Field(
        description="Classified action based on user input."
    )
    reasoning: str = Field(description="Brief explanation for the classified action.")

class LogActivityInput(BaseModel):
    """Structured output for extracting log activity details."""
    food_item: Optional[str] = Field(description="The food item to log, if any.")
    exercise_activity: Optional[str] = Field(description="The exercise activity to log, if any.")
    calories_consumed: Optional[int] = Field(description="The calories consumed, if provided directly.")
    calories_burned: Optional[int] = Field(description="The calories burned, if provided directly.")

class DataRetrievalClassifier(BaseModel):
    """Structured output for classifying data retrieval intent."""
    query_type: Literal["historical_data", "net_calories"] = Field(
        description="Classified query type for data retrieval."
    )
    reasoning: str = Field(description="Brief explanation for the classified query type.")

def data_logging(state: GraphState) -> Command[Literal["log_activity", "data_retrieval"]]:
    """
    Node purpose: Initial node to determine if the user wants to log data, retrieve historical data, or calculate net calories.
    Implementation reasoning: Uses structured output to classify the user's intent for routing.
    """
    structured_llm = llm.with_structured_output(ActionClassifier)
    user_message = state["messages"][-1].content
    
    prompt = f"""Classify the user's request: "{user_message}".
    Is the user trying to:
    - 'log_data': if they want to record food intake or exercise.
    - 'retrieve_historical_data': if they want to see past records of food or exercise.
    - 'calculate_net_calories': if they want to know their net calorie balance.
    """
    
    classification: ActionClassifier = structured_llm.invoke(prompt)
    
    if classification.action == "log_data":
        return Command(goto="log_activity", update={"action": classification.action, "messages": [AIMessage(content=f"Intent classified as: {classification.action}.")]})
    else:
        return Command(goto="data_retrieval", update={"action": classification.action, "messages": [AIMessage(content=f"Intent classified as: {classification.action}.")]})

log_activity_tools = [CreateCalorieData, SearchFoodCalorieAPI, SearchExerciseCalorieAPI]
def log_activity(state: GraphState) -> GraphState:
    """
    Node purpose: Logs food and exercise activities, calculating calorie intake and expenditure.
    Implementation reasoning: Uses a ReAct agent to intelligently use tools for logging and calorie lookup.
    """
    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: any

    class LogActivityResponse(BaseModel):
        food_item: Optional[str] = Field(description="The food item that was logged.")
        exercise_activity: Optional[str] = Field(description="The exercise activity that was logged.")
        calories_consumed: Optional[int] = Field(description="The calories consumed, if applicable.")
        calories_burned: Optional[int] = Field(description="The calories burned, if applicable.")
        status: str = Field(description="Status of the logging operation.")
        message: str = Field(description="A user-friendly message about the logging result.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an AI assistant that helps log food and exercise activities. Use the provided tools to search for calorie information and log data. If the user provides direct calorie numbers, use them. Otherwise, search for them using the tools. Always confirm the logging action and provide a summary.",
        tools=log_activity_tools,
        state_schema=CustomStateForReact,
        response_format=LogActivityResponse
    )

    user_message = state["messages"][-1].content
    
    # Extract initial logging details using structured output
    structured_llm_input = llm.with_structured_output(LogActivityInput)
    extracted_details: LogActivityInput = structured_llm_input.invoke(f"Extract food item, exercise activity, calories consumed, and calories burned from: {user_message}")

    # Prepare agent input, prioritizing extracted direct values
    agent_input_messages = [HumanMessage(content=user_message)]
    if extracted_details.food_item:
        agent_input_messages.append(HumanMessage(content=f"Food item: {extracted_details.food_item}"))
    if extracted_details.exercise_activity:
        agent_input_messages.append(HumanMessage(content=f"Exercise activity: {extracted_details.exercise_activity}"))
    if extracted_details.calories_consumed is not None:
        agent_input_messages.append(HumanMessage(content=f"Calories consumed: {extracted_details.calories_consumed}"))
    if extracted_details.calories_burned is not None:
        agent_input_messages.append(HumanMessage(content=f"Calories burned: {extracted_details.calories_burned}"))

    result: LogActivityResponse = agent.invoke({"messages": agent_input_messages})["structured_response"]

    return {
        "food_item": result.food_item,
        "exercise_activity": result.exercise_activity,
        "calories_consumed": result.calories_consumed,
        "calories_burned": result.calories_burned,
        "messages": [AIMessage(content=result.message)]
    }

def data_retrieval(state: GraphState) -> Command[Literal["retrieve_historical_data", "calculate_net_calories"]]:
    """
    Node purpose: Routes to either historical data retrieval or net calorie calculation based on user query.
    Implementation reasoning: Uses structured output to classify the specific type of data retrieval requested.
    """
    structured_llm = llm.with_structured_output(DataRetrievalClassifier)
    user_message = state["messages"][-1].content
    
    prompt = f"""Classify the user's data retrieval query: "{user_message}".
    Is the user trying to:
    - 'historical_data': if they want to see past records of food or exercise.
    - 'net_calories': if they want to know their net calorie balance.
    """
    
    classification: DataRetrievalClassifier = structured_llm.invoke(prompt)
    
    if classification.query_type == "historical_data":
        return Command(goto="retrieve_historical_data", update={"query_type": classification.query_type, "messages": [AIMessage(content=f"Query type classified as: {classification.query_type}.")]})
    else:
        return Command(goto="calculate_net_calories", update={"query_type": classification.query_type, "messages": [AIMessage(content=f"Query type classified as: {classification.query_type}.")]})

retrieve_historical_data_tools = [ReadCalorieData]
def retrieve_historical_data(state: GraphState) -> GraphState:
    """
    Node purpose: Retrieves and presents past food consumption and exercise data.
    Implementation reasoning: Uses a ReAct agent with the ReadCalorieData tool to fetch historical information.
    """
    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: any

    class HistoricalDataResponse(BaseModel):
        data: str = Field(description="The retrieved historical data.")
        message: str = Field(description="A user-friendly message summarizing the retrieved data.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an AI assistant that retrieves historical calorie data. Use the 'ReadCalorieData' tool to fetch past food consumption and exercise data based on the user's query. Summarize the retrieved data clearly.",
        tools=retrieve_historical_data_tools,
        state_schema=CustomStateForReact,
        response_format=HistoricalDataResponse
    )

    user_message = state["messages"][-1].content
    result: HistoricalDataResponse = agent.invoke({"messages": [HumanMessage(content=user_message)]})["structured_response"]

    return {
        "historical_data": result.data,
        "messages": [AIMessage(content=result.message)]
    }

calculate_net_calories_tools = [ReadCalorieData]
def calculate_net_calories(state: GraphState) -> GraphState:
    """
    Node purpose: Calculates and displays the net calorie balance over a specified period.
    Implementation reasoning: Uses a ReAct agent with the ReadCalorieData tool to fetch data and then calculates net calories.
    """
    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: any

    class NetCalorieResponse(BaseModel):
        net_calories: int = Field(description="The calculated net calorie balance.")
        message: str = Field(description="A user-friendly message displaying the net calorie balance.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an AI assistant that calculates net calorie balance. Use the 'ReadCalorieData' tool to fetch relevant calorie consumption and expenditure data. Then, calculate the net calorie balance and present it clearly.",
        tools=calculate_net_calories_tools,
        state_schema=CustomStateForReact,
        response_format=NetCalorieResponse
    )

    user_message = state["messages"][-1].content
    result: NetCalorieResponse = agent.invoke({"messages": [HumanMessage(content=user_message)]})["structured_response"]

    return {
        "net_calories": result.net_calories,
        "messages": [AIMessage(content=result.message)]
    }

workflow = StateGraph(GraphState)

workflow.add_node("data_logging", data_logging)
workflow.add_node("log_activity", log_activity)
workflow.add_node("data_retrieval", data_retrieval)
workflow.add_node("retrieve_historical_data", retrieve_historical_data)
workflow.add_node("calculate_net_calories", calculate_net_calories)

workflow.add_edge(START, "data_logging")

workflow.add_conditional_edges(
    "data_logging",
    lambda state: state["action"],
    {
        "log_data": "log_activity",
        "retrieve_historical_data": "data_retrieval",
        "calculate_net_calories": "data_retrieval"
    }
)

workflow.add_edge("log_activity", END)

workflow.add_conditional_edges(
    "data_retrieval",
    lambda state: state["query_type"],
    {
        "historical_data": "retrieve_historical_data",
        "net_calories": "calculate_net_calories"
    }
)

workflow.add_edge("retrieve_historical_data", END)
workflow.add_edge("calculate_net_calories", END)

checkpointer = InMemorySaver()
app = workflow.compile(
    checkpointer=checkpointer
)

## Required Keys and Credentials
# - OPENAI_API_KEY: For accessing OpenAI models.
# - USER_ID: For Composio tools (CreateCalorieData, ReadCalorieData).
# - NINJA_API_KEY: For the SearchExerciseCalorieAPI tool.
```
'''