nutrition_main = r'''
from typing import Dict, Any, List, Optional, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from tools_code import Food_Database_API_Search, Food_Database_API_Read, Exercise_Database_API_Search, Exercise_Database_API_Read, Database_Management_System_Create, Database_Management_System_Read, Database_Management_System_Update
import os

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(MessagesState):
    """ The GraphState represents the state of the LangGraph workflow. """
    user_input: str
    food_items: Optional[List[str]] = None
    exercise_activities: Optional[List[str]] = None
    calories_consumed: Optional[float] = None
    calories_burned: Optional[float] = None
    historical_query: Optional[str] = None
    net_calorie_query: Optional[str] = None
    storage_status: Optional[str] = None
    retrieved_data: Optional[str] = None
    net_calorie_balance: Optional[float] = None

class CalorieCalculationOutput(BaseModel):
    """Structured output for calorie calculation and intent routing."""
    intent: Literal["data_entry", "historical_query", "net_calorie_analysis", "unclear"] = Field(
        description="Classified intent of the user's query."
    )
    food_items: Optional[List[str]] = Field(
        default=None, description="List of food items mentioned in the user input."
    )
    exercise_activities: Optional[List[str]] = Field(
        default=None, description="List of exercise activities mentioned in the user input."
    )
    historical_query: Optional[str] = Field(
        default=None, description="Specific query for historical data, if intent is historical_query."
    )
    net_calorie_query: Optional[str] = Field(
        default=None, description="Specific query for net calorie analysis, if intent is net_calorie_analysis."
    )

calorie_calculation_tools = [Food_Database_API_Search, Food_Database_API_Read, Exercise_Database_API_Search, Exercise_Database_API_Read]
def calculate_calories(state: GraphState) -> GraphState:
    """
    Node purpose: Calculates calories from user-reported food intake and exercise activities.
                   This node also acts as a router, determining the next step based on the user's intent.
    Implementation reasoning: Uses a ReAct agent to leverage tools for calorie calculation and an LLM with structured output for intent classification and data extraction.
    """
    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: CalorieCalculationOutput

    agent = create_react_agent(
        model=llm,
        prompt="You are a helpful assistant that calculates calories from food and exercise. "
               "You can also identify if the user is asking for historical data or net calorie analysis. "
               "Use the provided tools to search for food and exercise calorie information. "
               "After processing, classify the user's intent and extract relevant entities.",
        tools=calorie_calculation_tools,
        state_schema=CustomStateForReact,
        response_format=CalorieCalculationOutput
    )

    user_message = state["messages"][-1].content
    result: CalorieCalculationOutput = agent.invoke({"messages": state["messages"]})["structured_response"]

    calories_consumed = 0.0
    if result.food_items:
        for food_item in result.food_items:
            search_result = Food_Database_API_Search(food_query=food_item)
            if search_result and search_result.get('parsed') and search_result['parsed'][0].get('food'):
                food_id = search_result['parsed'][0]['food']['foodId']
                read_result = Food_Database_API_Read(food_id=food_id)
                if read_result and read_result.get('totalNutrients') and read_result['totalNutrients'].get('ENERC_KCAL'):
                    calories_consumed += read_result['totalNutrients']['ENERC_KCAL']['quantity']

    calories_burned = 0.0
    if result.exercise_activities:
        # For simplicity, assuming a generic exercise calculation or a placeholder for actual exercise data
        # In a real scenario, this would involve more sophisticated parsing of exercise details (duration, intensity, etc.)
        # and using Exercise_Database_API_Search with appropriate parameters.
        # For now, we'll just use a dummy value or a simple calculation if parameters are available.
        # Example: if user input contains "ran for 30 minutes, 150 lbs, male, 30 years old, 140 bpm"
        # This would require more advanced NLP to extract these parameters.
        # For demonstration, we'll assume a fixed value or a simple prompt to the user for more details.
        # If Exercise_Database_API_Search requires specific parameters, the agent would need to extract them.
        # As Exercise_Database_API_Search requires specific parameters (gender, age, weight, heart_rate, time_hours),
        # and these are not directly extracted by the current structured output, we'll use a placeholder.
        # A more robust solution would involve the LLM extracting these details or prompting the user.
        calories_burned = 200.0 * len(result.exercise_activities) # Placeholder value

    return {
        "messages": [AIMessage(content=f"Intent classified as: {result.intent}. Calories consumed: {calories_consumed}, Calories burned: {calories_burned}")],
        "user_input": user_message,
        "food_items": result.food_items,
        "exercise_activities": result.exercise_activities,
        "calories_consumed": calories_consumed,
        "calories_burned": calories_burned,
        "historical_query": result.historical_query,
        "net_calorie_query": result.net_calorie_query
    }

data_storage_tools = [Database_Management_System_Create, Database_Management_System_Update]
def store_data(state: GraphState) -> GraphState:
    """
    Node purpose: Stores calculated calorie data for both food consumption and exercise into the database.
    Implementation reasoning: Uses a ReAct agent to interact with database management tools.
    """
    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: Any # The response from the database tool can be varied

    agent = create_react_agent(
        model=llm,
        prompt="You are an assistant that stores calorie data. Use the provided tools to create or update database entries.",
        tools=data_storage_tools,
        state_schema=CustomStateForReact,
        response_format=Any # No specific structured output needed for this node's return
    )

    storage_status = "failed"
    if state["calories_consumed"] is not None and state["calories_consumed"] > 0:
        # Assuming a simple structure for Notion database entry
        food_data = {
            "Food Items": {"rich_text": [{"text": {"content": ", ".join(state["food_items"]) if state["food_items"] else "N/A"}}]},
            "Calories Consumed": {"number": state["calories_consumed"]}
        }
        try:
            # This assumes a specific database ID for food entries. Replace with actual ID.
            db_id = os.environ.get("NOTION_FOOD_DATABASE_ID")
            if not db_id:
                raise ValueError("NOTION_FOOD_DATABASE_ID environment variable must be set.")
            Database_Management_System_Create(database_id=db_id, properties=food_data)
            storage_status = "food data stored"
        except Exception as e:
            storage_status = f"failed to store food data: {e}"

    if state["calories_burned"] is not None and state["calories_burned"] > 0:
        exercise_data = {
            "Exercise Activities": {"rich_text": [{"text": {"content": ", ".join(state["exercise_activities"]) if state["exercise_activities"] else "N/A"}}]},
            "Calories Burned": {"number": state["calories_burned"]}
        }
        try:
            # This assumes a specific database ID for exercise entries. Replace with actual ID.
            db_id = os.environ.get("NOTION_EXERCISE_DATABASE_ID")
            if not db_id:
                raise ValueError("NOTION_EXERCISE_DATABASE_ID environment variable must be set.")
            Database_Management_System_Create(database_id=db_id, properties=exercise_data)
            storage_status += " and exercise data stored" if storage_status != "failed" else "exercise data stored"
        except Exception as e:
            storage_status += f" and failed to store exercise data: {e}" if storage_status != "failed" else f"failed to store exercise data: {e}"

    if storage_status == "failed":
        storage_status = "No data to store or all storage attempts failed."

    return {
        "messages": [AIMessage(content=f"Data storage status: {storage_status}")],
        "storage_status": storage_status
    }

historical_retrieval_tools = [Database_Management_System_Read]
def retrieve_historical_data(state: GraphState) -> GraphState:
    """
    Node purpose: Retrieves and presents historical data on food consumption and exercise based on user queries.
    Implementation reasoning: Uses a ReAct agent to interact with the database read tool.
    """
    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: Any

    agent = create_react_agent(
        model=llm,
        prompt="You are an assistant that retrieves historical calorie data. Use the provided tool to query the database based on the user's request.",
        tools=historical_retrieval_tools,
        state_schema=CustomStateForReact,
        response_format=Any
    )

    query = state["historical_query"]
    retrieved_data = "No historical data found."
    if query:
        try:
            # This assumes a generic database ID for historical queries. In a real app, you might query specific food/exercise DBs.
            db_id = os.environ.get("NOTION_MASTER_DATABASE_ID") # Or specific food/exercise DB IDs
            if not db_id:
                raise ValueError("NOTION_MASTER_DATABASE_ID environment variable must be set.")
            # The query parameter for Notion's query_database tool is complex, often requiring a filter object.
            # For simplicity, we'll assume a basic query or prompt the user for more specific filters.
            # A more advanced implementation would involve the LLM constructing the Notion filter object.
            # Example: Database_Management_System_Read(database_id=db_id, filter={"property": "Date", "date": {"on_or_before": "2023-01-01"}})
            # For now, we'll just fetch some data and let the LLM summarize.
            response = Database_Management_System_Read(database_id=db_id)
            if response and response.get('results'):
                retrieved_data = "Retrieved historical data:\n"
                for item in response['results']:
                    props = item.get('properties', {})
                    food = props.get('Food Items', {}).get('rich_text', [{}])[0].get('text', {}).get('content', 'N/A')
                    consumed = props.get('Calories Consumed', {}).get('number', 'N/A')
                    exercise = props.get('Exercise Activities', {}).get('rich_text', [{}])[0].get('text', {}).get('content', 'N/A')
                    burned = props.get('Calories Burned', {}).get('number', 'N/A')
                    retrieved_data += f"- Food: {food}, Consumed: {consumed}, Exercise: {exercise}, Burned: {burned}\n"
            else:
                retrieved_data = "No historical data found for your query."
        except Exception as e:
            retrieved_data = f"Failed to retrieve historical data: {e}"

    return {
        "messages": [AIMessage(content=f"Historical data retrieval status: {retrieved_data}")],
        "retrieved_data": retrieved_data
    }

net_calorie_analysis_tools = [Database_Management_System_Read]
def analyze_net_calories(state: GraphState) -> GraphState:
    """
    Node purpose: Analyzes and provides insights into the overall calorie balance (consumed vs. burned) for a specified period.
    Implementation reasoning: Uses a ReAct agent to retrieve data and then an LLM to perform the analysis.
    """
    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: Any

    agent = create_react_agent(
        model=llm,
        prompt="You are an assistant that analyzes net calorie balance. Use the provided tool to retrieve data and then calculate the net balance.",
        tools=net_calorie_analysis_tools,
        state_schema=CustomStateForReact,
        response_format=Any
    )

    query = state["net_calorie_query"]
    net_calorie_balance = 0.0
    analysis_message = "Could not perform net calorie analysis."

    if query:
        try:
            db_id = os.environ.get("NOTION_MASTER_DATABASE_ID") # Or specific food/exercise DB IDs
            if not db_id:
                raise ValueError("NOTION_MASTER_DATABASE_ID environment variable must be set.")
            # Similar to historical retrieval, this would need a more sophisticated filter for time periods.
            response = Database_Management_System_Read(database_id=db_id)
            
            total_consumed = 0.0
            total_burned = 0.0
            if response and response.get('results'):
                for item in response['results']:
                    props = item.get('properties', {})
                    consumed = props.get('Calories Consumed', {}).get('number', 0.0)
                    burned = props.get('Calories Burned', {}).get('number', 0.0)
                    total_consumed += consumed
                    total_burned += burned
                net_calorie_balance = total_consumed - total_burned
                analysis_message = f"For your query '{query}', total calories consumed: {total_consumed}, total calories burned: {total_burned}. Net calorie balance: {net_calorie_balance}."
            else:
                analysis_message = "No data found for net calorie analysis."
        except Exception as e:
            analysis_message = f"Failed to perform net calorie analysis: {e}"

    return {
        "messages": [AIMessage(content=analysis_message)],
        "net_calorie_balance": net_calorie_balance
    }

def route_calorie_calculation(state: GraphState) -> Literal["data_storage", "historical_retrieval", "net_calorie_analysis", END]:
    """
    Routing function for calorie_calculation node.
    Determines the next node based on the user's intent.
    """
    if state["historical_query"]:
        return "historical_retrieval"
    elif state["net_calorie_query"]:
        return "net_calorie_analysis"
    elif (state["calories_consumed"] is not None and state["calories_consumed"] > 0) or \
         (state["calories_burned"] is not None and state["calories_burned"] > 0):
        return "data_storage"
    else:
        return END # Or a node to handle unclear intent

workflow = StateGraph(GraphState)

workflow.add_node("calorie_calculation", calculate_calories)
workflow.add_node("data_storage", store_data)
workflow.add_node("historical_retrieval", retrieve_historical_data)
workflow.add_node("net_calorie_analysis", analyze_net_calories)

workflow.add_edge(START, "calorie_calculation")

workflow.add_conditional_edges(
    "calorie_calculation",
    route_calorie_calculation,
    {
        "data_storage": "data_storage",
        "historical_retrieval": "historical_retrieval",
        "net_calorie_analysis": "net_calorie_analysis",
        END: END
    }
)

workflow.add_edge("data_storage", END)
workflow.add_edge("historical_retrieval", END)
workflow.add_edge("net_calorie_analysis", END)

checkpointer = InMemorySaver()
app = workflow.compile(
    checkpointer=checkpointer
)

## Required Keys and Credentials
# Environment variables:
# - OPENAI_API_KEY: For OpenAI LLM access.
# - EDAMAM_APP_ID: Application ID for Edamam Food Database API.
# - EDAMAM_APP_KEY: Application Key for Edamam Food Database API.
# - USER_ID: User ID for Composio tools (e.g., Notion).
# - NOTION_FOOD_DATABASE_ID: Notion database ID for storing food entries.
# - NOTION_EXERCISE_DATABASE_ID: Notion database ID for storing exercise entries.
# - NOTION_MASTER_DATABASE_ID: Notion database ID for general historical queries (can be the same as food/exercise or a combined one).
'''