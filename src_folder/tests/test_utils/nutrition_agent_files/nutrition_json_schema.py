json_schema_nutrition = r'''
{
    "justification": "The \"Plan and Execute\" architecture is suitable here because the agent needs to first interpret the user's intent (plan) and then execute the appropriate action (log data, retrieve historical data, or calculate net calories). The initial \"data_logging\" node acts as a router to direct the flow based on the user's request. This allows for a clear separation of concerns and efficient handling of different types of queries.",
    "nodes": [
      {
        "id": "data_logging",
        "schema_info": "GraphState:\n    type: TypedDict\n    fields:\n        - name: input\n          type: str\n        - name: action\n          type: str",
        "input_schema": "GraphState",
        "output_schema": "GraphState",
        "description": "Initial node to determine if the user wants to log data, retrieve historical data, or calculate net calories.",
        "function_name": "data_logging"
      },
      {
        "id": "log_activity",
        "schema_info": "GraphState:\n    type: TypedDict\n    fields:\n        - name: input\n          type: str\n        - name: food_item\n          type: str\n        - name: exercise_activity\n          type: str\n        - name: calories_consumed\n          type: int\n        - name: calories_burned\n          type: int",
        "input_schema": "GraphState",
        "output_schema": "GraphState",
        "description": "Logs food and exercise activities, calculating calorie intake and expenditure.",
        "function_name": "log_activity"
      },
      {
        "id": "data_retrieval",
        "schema_info": "GraphState:\n    type: TypedDict\n    fields:\n        - name: input\n          type: str\n        - name: query_type\n          type: str",
        "input_schema": "GraphState",
        "output_schema": "GraphState",
        "description": "Routes to either historical data retrieval or net calorie calculation based on user query.",
        "function_name": "data_retrieval"
      },
      {
        "id": "retrieve_historical_data",
        "schema_info": "GraphState:\n    type: TypedDict\n    fields:\n        - name: input\n          type: str\n        - name: historical_data\n          type: dict",
        "input_schema": "GraphState",
        "output_schema": "GraphState",
        "description": "Retrieves and presents past food consumption and exercise data.",
        "function_name": "retrieve_historical_data"
      },
      {
        "id": "calculate_net_calories",
        "schema_info": "GraphState:\n    type: TypedDict\n    fields:\n        - name: input\n          type: str\n        - name: net_calories\n          type: int",
        "input_schema": "GraphState",
        "output_schema": "GraphState",
        "description": "Calculates and displays the net calorie balance over a specified period.",
        "function_name": "calculate_net_calories"
      }
    ],
    "edges": [
      {
        "source": "__START__",
        "target": "data_logging",
        "routing_conditions": "",
        "conditional": false
      },
      {
        "source": "data_logging",
        "target": "log_activity",
        "routing_conditions": "If the user's input indicates a need to log food or exercise data.",
        "conditional": true
      },
      {
        "source": "data_logging",
        "target": "data_retrieval",
        "routing_conditions": "If the user's input indicates a need to retrieve historical data or calculate net calories.",
        "conditional": true
      },
      {
        "source": "log_activity",
        "target": "__END__",
        "routing_conditions": "",
        "conditional": false
      },
      {
        "source": "data_retrieval",
        "target": "retrieve_historical_data",
        "routing_conditions": "If the user's input is a query about historical data.",
        "conditional": true
      },
      {
        "source": "data_retrieval",
        "target": "calculate_net_calories",
        "routing_conditions": "If the user's input is a query about net calorie intake.",
        "conditional": true
      },
      {
        "source": "retrieve_historical_data",
        "target": "__END__",
        "routing_conditions": "",
        "conditional": false
      },
      {
        "source": "calculate_net_calories",
        "target": "__END__",
        "routing_conditions": "",
        "conditional": false
      }
    ],
    "tools": [
      {
        "name": "CreateCalorieData",
        "description": "Tool to create new calorie-related data entries.",
        "is_composio_tool": true,
        "composio_toolkit_slug": "Notion",
        "composio_tool_slug": "NOTION_INSERT_ROW_DATABASE",
        "node_ids": [
          "log_activity"
        ]
      },
      {
        "name": "ReadCalorieData",
        "description": "Tool to read existing calorie-related data.",
        "is_composio_tool": true,
        "composio_toolkit_slug": "Notion",
        "composio_tool_slug": "NOTION_QUERY_DATABASE",
        "node_ids": [
          "retrieve_historical_data",
          "calculate_net_calories"
        ]
      },
      {
        "name": "SearchFoodCalorieAPI",
        "description": "Tool to search for calorie information of various food items from an external database.",
        "is_composio_tool": false,
        "py_code": "# Install the SDK\n# pip install openfoodfacts\n\nfrom openfoodfacts import OpenFoodFactsAPI\n\napi = OpenFoodFactsAPI()\n\n# Search for a product by name\nproducts = api.search('apple')\nfor product in products:\n    print(product['product_name'], product.get('nutriments', {}))",
        "node_ids": [
          "log_activity"
        ]
      },
      {
        "name": "SearchExerciseCalorieAPI",
        "description": "Tool to search for calorie expenditure data for different types of exercises and durations.",
        "is_composio_tool": false,
        "py_code": "import requests\n\nactivity = 'skiing'\napi_url = f'https://api.api-ninjas.com/v1/caloriesburned?activity={activity}'\nheaders = {'X-Api-Key': 'YOUR_API_KEY'}\nresponse = requests.get(api_url, headers=headers)\nif response.status_code == 200:\n    print(response.json())\nelse:\n    print(f'Error: {response.status_code} - {response.text}')",
        "node_ids": [
          "log_activity"
        ]
      }
    ]
  }'''
