# code stub
json_dict = """{
  "nodes": {
    "__START__": {
      "schema_info": "None",
      "input_schema": "None",
      "output_schema": "None",
      "description": "The starting point of the graph.",
      "function_name": "start_node"
    },
    "supervisor": {
      "schema_info": "Router: TypedDict with next as a string literal of options.",
      "input_schema": "State",
      "output_schema": "State",
      "description": "This node manages the conversation between workers and decides which worker to route to next.",
      "function_name": "supervisor_node"
    },
    "food_logger": {
      "schema_info": "State: TypedDict with messages as a list of messages.",
      "input_schema": "State",
      "output_schema": "State",
      "description": "This node logs food intake and calculates calories.",
      "function_name": "food_logger_node"
    },
    "workout_tracker": {
      "schema_info": "State: TypedDict with messages as a list of messages.",
      "input_schema": "State",
      "output_schema": "State",
      "description": "This node logs workout details and calculates calories burnt.",
      "function_name": "workout_tracker_node"
    },
    "__END__": {
      "schema_info": "None",
      "input_schema": "None",
      "output_schema": "None",
      "description": "The endpoint of the graph, indicating completion.",
      "function_name": "end_node"
    }
  },
  "edges": {
    { 
      "source": "__START__", 
      "target": "supervisor", 
      "routing_conditions": "Start the process by routing to the supervisor node.", 
      "conditional": False 
    },
    { 
      "source": "supervisor", 
      "target": "food_logger", 
      "routing_conditions": "If the supervisor decides to route to food_logger based on user input.", 
      "conditional": True 
    },
    { 
      "source": "supervisor", 
      "target": "workout_tracker", 
      "routing_conditions": "If the supervisor decides to route to workout_tracker based on user input.", 
      "conditional": True 
    },
    { 
      "source": "supervisor", 
      "target": "__END__", 
      "routing_conditions": "If the supervisor decides to finish the process.", 
      "conditional": True
    },
    { 
      "source": "food_logger", 
      "target": "supervisor", 
      "routing_conditions": "After logging food intake, return control to the supervisor.", 
      "conditional": False 
    },
    { 
      "source": "workout_tracker", 
      "target": "supervisor", 
      "routing_conditions": "After logging workout details, return control to the supervisor.", 
      "conditional": False 
    }
  }
}
"""

code = """ 
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
model = ChatOpenAI(model="gpt-4o-mini")

# Create specialized agents

@tool
def food_logger_tool(food: str) -> str:
    \"\"\"Log food intake in the database\"\"\"
    # Parse food input and get nutrition data from Nutritionix API
    import requests
    import json
    from pymongo import MongoClient
    from datetime import datetime

    # Nutritionix API credentials should be in environment variables
    headers = {
        "x-app-id": os.getenv("NUTRITIONIX_APP_ID"),
        "x-app-key": os.getenv("NUTRITIONIX_API_KEY"),
        "x-remote-user-id": "0"  # 0 for development
    }

    # Query Nutritionix API
    endpoint = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    payload = {"query": food}
    
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        nutrition_data = response.json()

        # Connect to MongoDB
        client = MongoClient(os.getenv("MONGODB_URI"))
        db = client.nutrition_db
        food_logs = db.food_logs

        # Prepare document for MongoDB
        food_entry = {
            "food_name": food,
            "timestamp": datetime.now(),
            "nutrition_data": nutrition_data["foods"][0],
            "calories": nutrition_data["foods"][0]["nf_calories"],
            "protein": nutrition_data["foods"][0]["nf_protein"],
            "carbohydrates": nutrition_data["foods"][0]["nf_total_carbohydrate"],
            "fat": nutrition_data["foods"][0]["nf_total_fat"]
        }

        # Insert into MongoDB
        food_logs.insert_one(food_entry)
        client.close()

    except Exception as e:
        return f"Error logging food: {str(e)}"

@tool
def workout_tracker_tool(workout: str) -> str:
    \"\"\"Log workout details in the database\"\"\"
    # Parse workout input and get workout details from Google Fit API
    import requests
    import json
    from pymongo import MongoClient
    from datetime import datetime

    # Google Fit API credentials should be in environment variables
    headers = {
        "Authorization": f"Bearer {os.getenv('GOOGLE_FIT_TOKEN')}"
    }

    # Query Google Fit API
    endpoint = "https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate"
    
    try:
        # Connect to MongoDB
        client = MongoClient(os.getenv("MONGODB_URI"))
        db = client.fitness_db
        workout_logs = db.workout_logs

        # Prepare document for MongoDB
        workout_entry = {
            "workout_type": workout,
            "timestamp": datetime.now(),
            "duration_minutes": 30,  # This would come from Google Fit API
            "calories_burned": 250,  # This would come from Google Fit API
            "heart_rate_avg": 140,   # This would come from Google Fit API
            "steps": 4000            # This would come from Google Fit API
        }

        # Insert into MongoDB
        workout_logs.insert_one(workout_entry)
        client.close()
        
        return f"Successfully logged workout: {workout}"

    except Exception as e:
        return f"Error logging workout: {str(e)}"

food_logger_agent = create_react_agent(
    model=model,
    tools=[food_logger_tool],
    name="food_logger",
    prompt="user will tell you what they ate and you will log it in the database"
)

workout_tracker_agent = create_react_agent(
    model=model,
    tools=[workout_tracker_tool],
    name="workout_tracker",
    prompt="user will tell you what they did and you will log it in the database"
)

# Create supervisor workflow
workflow = create_supervisor(
    [food_logger_agent, workout_tracker_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For tracking calories, use food_logger "
        "For tracking workout, use workout_tracker"
    )
)

# Compile and run
app = workflow.compile()
"""