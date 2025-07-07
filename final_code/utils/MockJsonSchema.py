json_schema_str = """
{{
  "justification": "A standard stateful graph construction is chosen because the problem involves distinct, sequential steps based on user input, without requiring complex planning, hierarchical delegation, or iterative refinement. Each use case can be directly mapped to a specific node, and a router can effectively direct the flow.",
  "nodes": [
    {{
      "id": "route_query",
      "schema_info": "GraphState:\n  type: TypedDict\n  fields:\n    - name: user_input\n      type: str\n      description: The user's original query.\n    - name: intent\n      type: str\n      description: The identified intent of the user's query (e.g., 'track_food', 'track_exercise', 'get_diet_history', 'get_exercise_history', 'get_net_effect').",
      "input_schema": "GraphState",
      "output_schema": "GraphState",
      "description": "Determines the intent of the user's query (e.g., calorie tracking for food/exercise, historical Q&A, net effect Q&A) and routes to the appropriate node.",
      "function_name": "route_query"
    }},
    {{
      "id": "track_food_calories",
      "schema_info": "GraphState:\n  type: TypedDict\n  fields:\n    - name: user_input\n      type: str\n    - name: food_items\n      type: list[dict]\n      description: A list of dictionaries, each containing 'item' (str) and 'quantity' (str).\n    - name: calculated_calories\n      type: int\n      description: The total calculated calories for the food intake.\n    - name: storage_status\n      type: str\n      description: Status of data storage (e.g., 'success', 'failure').",
      "input_schema": "GraphState",
      "output_schema": "GraphState",
      "description": "Calculates and stores calories consumed from user-reported food intake. It extracts food items and quantities from the user input, looks up calorie information, and stores it in the database.",
      "function_name": "track_food_calories"
    }},
    {{
      "id": "track_exercise_calories",
      "schema_info": "GraphState:\n  type: TypedDict\n  fields:\n    - name: user_input\n      type: str\n    - name: exercise_details\n      type: dict\n      description: A dictionary containing 'type' (str) and 'duration_minutes' (int).\n    - name: estimated_calories_burned\n      type: int\n      description: The estimated calories burned during the exercise.\n    - name: storage_status\n      type: str\n      description: Status of data storage (e.g., 'success', 'failure').",
      "input_schema": "GraphState",
      "output_schema": "GraphState",
      "description": "Calculates and stores calories burned from user-reported exercise. It extracts exercise type and duration from the user input, estimates calorie expenditure, and stores it in the database.",
      "function_name": "track_exercise_calories"
    }},
    {{
      "id": "get_diet_history",
      "schema_info": "GraphState:\n  type: TypedDict\n  fields:\n    - name: user_input\n      type: str\n    - name: query_date_range\n      type: str\n      description: The date or date range for the historical query (e.g., 'yesterday', 'last week', 'Monday').\n    - name: diet_history_results\n      type: list[dict]\n      description: A list of dictionaries, each representing a food entry with 'date', 'item', and 'calories'.",
      "input_schema": "GraphState",
      "output_schema": "GraphState",
      "description": "Retrieves and summarizes past food consumption data based on user queries (e.g., 'What did I eat yesterday?'). It queries the database for relevant entries.",
      "function_name": "get_diet_history"
    }},
    {{
      "id": "get_exercise_history",
      "schema_info": "GraphState:\n  type: TypedDict\n  fields:\n    - name: user_input\n      type: str\n    - name: query_date_range\n      type: str\n      description: The date or date range for the historical query.\n    - name: exercise_history_results\n      type: list[dict]\n      description: A list of dictionaries, each representing an exercise entry with 'date', 'type', 'duration_minutes', and 'calories_burned'.",
      "input_schema": "GraphState",
      "output_schema": "GraphState",
      "description": "Retrieves and summarizes past exercise data based on user queries (e.g., 'How much did I run last week?'). It queries the database for relevant entries.",
      "function_name": "get_exercise_history"
    }},
    {{
      "id": "get_net_calorie_effect",
      "schema_info": "GraphState:\n  type: TypedDict\n  fields:\n    - name: user_input\n      type: str\n    - name: query_period\n      type: str\n      description: The period for the net calorie calculation (e.g., 'today', 'this week', 'last month').\n    - name: total_calories_consumed\n      type: int\n    - name: total_calories_burned\n      type: int\n    - name: net_calorie_balance\n      type: int",
      "input_schema": "GraphState",
      "output_schema": "GraphState",
      "description": "Calculates and provides insights into the overall calorie balance (consumed vs. burned) for a specified period. It queries the database for both food and exercise entries.",
      "function_name": "get_net_calorie_effect"
    }}
  ],
  "edges": [
    {{
      "source": "__START__",
      "target": "route_query",
      "routing_conditions": "Always proceeds to the 'route_query' node.",
      "conditional": false
    }},
    {{
      "source": "route_query",
      "target": "track_food_calories",
      "routing_conditions": "If the query is about calorie tracking for food, route to 'track_food_calories'.",
      "conditional": true
    }},
    {{
      "source": "route_query",
      "target": "track_exercise_calories",
      "routing_conditions": "If the query is about calorie tracking for exercise, route to 'track_exercise_calories'.",
      "conditional": true
    }},
    {{
      "source": "route_query",
      "target": "get_diet_history",
      "routing_conditions": "If the query is about historical diet information, route to 'get_diet_history'.",
      "conditional": true
    }},
    {{
      "source": "route_query",
      "target": "get_exercise_history",
      "routing_conditions": "If the query is about historical exercise information, route to 'get_exercise_history'.",
      "conditional": true
    }},
    {{
      "source": "route_query",
      "target": "get_net_calorie_effect",
      "routing_conditions": "If the query is about net calorie effect, route to 'get_net_calorie_effect'.",
      "conditional": true
    }},
    {{
      "source": "track_food_calories",
      "target": "__END__",
      "routing_conditions": "After tracking food calories, the process ends.",
      "conditional": false
    }},
    {{
      "source": "track_exercise_calories",
      "target": "__END__",
      "routing_conditions": "After tracking exercise calories, the process ends.",
      "conditional": false
    }},
    {{
      "source": "get_diet_history",
      "target": "__END__",
      "routing_conditions": "After retrieving diet history, the process ends.",
      "conditional": false
    }},
    {{
      "source": "get_exercise_history",
      "target": "__END__",
      "routing_conditions": "After retrieving exercise history, the process ends.",
      "conditional": false
    }},
    {{
      "source": "get_net_calorie_effect",
      "target": "__END__",
      "routing_conditions": "After retrieving net calorie effect, the process ends.",
      "conditional": false
    }}
  ],
  "tools": [
    {{
      "name": "calorie_lookup_tool",
      "description": "A tool to look up calorie information for various food items. Input: food item (str). Output: estimated calories (int).",
      "is_composio_tool": false,
      "composio_toolkit_slug": null,
      "composio_tool_slug": null,
      "py_code": "def calorie_lookup_tool(food_item: str) -> int:\n    # This would ideally call an external API or a local database\n    food_calories = {{\n        \"banana\": 105,\n        \"apple\": 95,\n        \"chicken breast\": 165, # per 100g\n        \"rice\": 130 # per 100g cooked\n    }}\n    return food_calories.get(food_item.lower(), 0)",
      "node_ids": [
        "track_food_calories"
      ]
    }},
    {{
      "name": "exercise_calorie_estimator_tool",
      "description": "A tool to estimate calories burned for various exercise types and durations. Input: exercise type (str), duration in minutes (int). Output: estimated calories burned (int).",
      "is_composio_tool": false,
      "composio_toolkit_slug": null,
      "composio_tool_slug": null,
      "py_code": "def exercise_calorie_estimator_tool(exercise_type: str, duration_minutes: int) -> int:\n    # This would ideally use METs (Metabolic Equivalents of Task) or other formulas\n    # For simplicity, using rough estimates\n    exercise_calories_per_minute = {{\n        \"running\": 10,\n        \"walking\": 5,\n        \"swimming\": 7,\n        \"cycling\": 8\n    }}\n    return exercise_calories_per_minute.get(exercise_type.lower(), 0) * duration_minutes",
      "node_ids": [
        "track_exercise_calories"
      ]
    }},
    {{
      "name": "store_food_intake_tool",
      "description": "A tool to store calorie intake data (food items and calories) into a database. Input: date (str), food item (str), calories (int). Output: success/failure status (str).",
      "is_composio_tool": false,
      "composio_toolkit_slug": null,
      "composio_tool_slug": null,
      "py_code": "def store_food_intake_tool(date: str, food_item: str, calories: int) -> str:\n    # Simulate database storage\n    print(f\"Storing food intake: Date: {{date}}, Item: {{food_item}}, Calories: {{calories}}\")\n    return \"success\"",
      "node_ids": [
        "track_food_calories"
      ]
    }},
    {{
      "name": "store_exercise_data_tool",
      "description": "A tool to store exercise data (type, duration, calories burned) into a database. Input: date (str), exercise type (str), duration minutes (int), calories burned (int). Output: success/failure status (str).",
      "is_composio_tool": false,
      "composio_toolkit_slug": null,
      "composio_tool_slug": null,
      "py_code": "def store_exercise_data_tool(date: str, exercise_type: str, duration_minutes: int, calories_burned: int) -> str:\n    # Simulate database storage\n    print(f\"Storing exercise data: Date: {{date}}, Type: {{exercise_type}}, Duration: {{duration_minutes}} mins, Calories Burned: {{calories_burned}}\")\n    return \"success\"",
      "node_ids": [
        "track_exercise_calories"
      ]
    }},
    {{
      "name": "get_food_history_tool",
      "description": "A tool to retrieve historical food consumption data from the database based on a date or date range. Input: query_date_range (str). Output: list of food entries (list[dict]).",
      "is_composio_tool": false,
      "composio_toolkit_slug": null,
      "composio_tool_slug": null,
      "py_code": "def get_food_history_tool(query_date_range: str) -> list[dict]:\n    # Simulate database retrieval\n    if query_date_range == \"yesterday\":\n        return [{{\"date\": \"2023-10-26\", \"item\": \"banana\", \"calories\": 105}}]\n    elif query_date_range == \"Monday\":\n        return [{{\"date\": \"2023-10-23\", \"item\": \"sandwich\", \"calories\": 400}}, {{\"date\": \"2023-10-23\", \"item\": \"apple\", \"calories\": 95}}]\n    return []",
      "node_ids": [
        "get_diet_history",
        "get_net_calorie_effect"
      ]
    }},
    {{
      "name": "get_exercise_history_tool",
      "description": "A tool to retrieve historical exercise data from the database based on a date or date range. Input: query_date_range (str). Output: list of exercise entries (list[dict]).",
      "is_composio_tool": false,
      "composio_toolkit_slug": null,
      "composio_tool_slug": null,
      "py_code": "def get_exercise_history_tool(query_date_range: str) -> list[dict]:\n    # Simulate database retrieval\n    if query_date_range == \"yesterday\":\n        return [{{\"date\": \"2023-10-26\", \"type\": \"running\", \"duration_minutes\": 30, \"calories_burned\": 350}}]\n    elif query_date_range == \"last week\":\n        return [{{\"date\": \"2023-10-23\", \"type\": \"running\", \"duration_minutes\": 20, \"calories_burned\": 250}}, {{\"date\": \"2023-10-25\", \"type\": \"walking\", \"duration_minutes\": 60, \"calories_burned\": 300}}]\n    return []",
      "node_ids": [
        "get_exercise_history",
        "get_net_calorie_effect"
      ]
    }}
  ]
}}
"""