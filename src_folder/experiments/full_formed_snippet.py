nutrition_str = """ 
from typing import TypedDict, Literal
from dataclasses import dataclass
from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
import json

# Define the possible nodes for routing
Worker = Literal['food_logger', 'workout_tracker', '__END__']

# State schema for supervisor node
class State(TypedDict):
    user_input: str

# Supervisor node implementation using langchain LLM for decision making
def supervisor(state: State) -> Command[Worker]:
    \"\"\"
    Node function that decides which worker node to route to next based on user input.
    Returns a Command with the 'goto' field set to the next node name.
    \"\"\"
    user_input = state.get('user_input', '').strip().lower()

    # Map user input to next node names
    input_to_worker = {
        'log food': 'food_logger',
        'log workout': 'workout_tracker',
        'finish': '__END__',
        'end': '__END__',
        'quit': '__END__',
    }

    next_node = input_to_worker.get(user_input, '__END__')

    # Return Command to route to the next node without state update
    return Command(goto=next_node)

# State schema for food_logger and workout_tracker nodes
@dataclass
class MessagesState:
    next: str

StateMessages = MessagesState

# food_logger node implementation
def food_logger_node(state: StateMessages) -> Literal['supervisor']:
    \"\"\"
    This node logs food intake and calculates calories.
    Since the input is a State object with a 'next' string,
    we assume 'next' contains information about the food intake.
    The function logs the intake and calculates calories,
    then returns the command 'supervisor'.
    \"\"\"
    food_entries = state.next.split(',')
    total_calories = 0
    for entry in food_entries:
        try:
            food, cal_str = entry.split(':')
            calories = int(cal_str)
            print(f"Logged food: {food.strip()}, Calories: {{calories}}")
            total_calories += calories
        except ValueError:
            print(f"Skipping invalid entry: {{entry}}")

    print(f"Total calories consumed: {{total_calories}}")

    return 'supervisor'

# workout_tracker node implementation
def workout_tracker_node(state: StateMessages) -> Literal['supervisor']:
    \"\"\"
    This node logs workouts and calculates calories burned.
    Given the state, it processes the workout data and returns the command 'supervisor'.
    \"\"\"
    try:
        workout_data = state.next.strip().split(',')
        if len(workout_data) != 3:
            raise ValueError("Invalid workout data format. Expected 'type,duration,weight'")

        workout_type, duration_str, weight_str = workout_data
        duration = float(duration_str)
        weight = float(weight_str)

        # Simple MET values for example workouts
        met_values = {
            'running': 9.8,
            'cycling': 7.5,
            'walking': 3.8,
            'swimming': 8.0,
            'yoga': 3.0
        }

        met = met_values.get(workout_type.lower(), 5.0)  # default MET if unknown

        # Calories burned formula: Calories = MET * weight_kg * duration_hours
        calories_burned = met * weight * (duration / 60)

        print(f"Workout logged: {{workout_type}} for {{duration}} minutes at {{weight}} kg")
        print(f"Calories burned: {{calories_burned:.2f}}")

    except Exception as e:
        print(f"Error processing workout data: {{e}}")

    return 'supervisor'

# Create the graph instance
graph = Graph()

# Add nodes with their implementations
graph.add_node("supervisor", supervisor)
graph.add_node("food_logger", food_logger_node)
graph.add_node("workout_tracker", workout_tracker_node)

# Add edges according to the graph definition

# edge_1: __START__ -> supervisor (non-conditional)
graph.add_edge("__START__", "supervisor")

# edge_2, edge_3, edge_4: supervisor -> food_logger/workout_tracker/__END__ (conditional edges)
graph.add_conditional_edges("supervisor", supervisor)

# edge_5: food_logger -> supervisor (non-conditional)
graph.add_edge("food_logger", "supervisor")

# edge_6: workout_tracker -> supervisor (non-conditional)
graph.add_edge("workout_tracker", "supervisor")

# Assign an InMemoryCheckpointer to the graph
graph.checkpointer = InMemoryCheckpointer()

# Assign the final graph to the variable final_app
final_app = graph.compile(checkpointer = InMemoryCheckpointer())

print(final_app.get_graph(xray=True).to_json())

with open("/home/user/sample.json" , "w" ) as write:
    json.dump(final_app.get_graph(xray=True).to_json(), write)
"""

python_code_snippet = """
import os
import json
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing import List, Annotated, TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Literal
import warnings
warnings.filterwarnings("ignore")
print("Running code with warnings suppressed.")

os.environ["TAVILY_API_KEY"] = "your_tavily_api_key_here"
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"


class State(MessagesState):
    next: str

# Tool for searching diet and exercise information
diet_exercise_tool = TavilySearchResults(max_results=5)

@tool
def log_meal(meal_description: str) -> str:
    "Log the meal described by the user."
    # Here we would implement the logic to log the meal
    return "Meal logged: " + meal_description


### Step 2: Create Agent Nodes

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o")

# Diet Agent
diet_agent = create_react_agent(llm, tools=[diet_exercise_tool])

def diet_node(state: State) -> Command[Literal["supervisor"]]:
    result = diet_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="diet_agent")
            ]
        },
        goto="supervisor",
    )

# Exercise Agent
exercise_agent = create_react_agent(llm, tools=[diet_exercise_tool])

def exercise_node(state: State) -> Command[Literal["supervisor"]]:
    result = exercise_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="exercise_agent")
            ]
        },
        goto="supervisor",
    )

# Logging Agent
logging_agent = create_react_agent(llm, tools=[log_meal])

def logging_node(state: State) -> Command[Literal["supervisor"]]:
    result = logging_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="logging_agent")
            ]
        },
        goto="supervisor",
    )

### Step 3: Create Supervisor Node

def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        "Worker to route to next. If no workers needed, route to FINISH."
        next: Literal["FINISH", "diet_agent", "exercise_agent", "logging_agent"]
        
    def supervisor_node(state: State) -> Command[Literal["diet_agent", "exercise_agent", "logging_agent", "__end__"]]:
        "An LLM-based router."
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state.messages
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node


teams_supervisor_node = make_supervisor_node(llm, ["diet_agent", "exercise_agent", "logging_agent"])

### Step 4: Build the StateGraph

# Now we will build the `StateGraph` for the `Fitness Advisor` team.

# Create the graph
fitness_builder = StateGraph(State)
fitness_builder.add_node("supervisor", teams_supervisor_node)
fitness_builder.add_node("diet_agent", diet_node)
fitness_builder.add_node("exercise_agent", exercise_node)
fitness_builder.add_node("logging_agent", logging_node)

fitness_builder.add_edge(START, "supervisor")
fitness_graph = fitness_builder.compile()

print(fitness_graph.get_graph(xray=True).to_json())

with open("/home/user/sample.json" , "w" ) as write:
    json.dump(fitness_graph.get_graph(xray=True).to_json(), write)
"""