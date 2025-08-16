nutrition_tools_code= r'''
from composio import Composio
from composio_langchain import LangchainProvider
from langchain_core.tools import tool
import os
import requests
from openfoodfacts import OpenFoodFactsAPI

composio = Composio(provider=LangchainProvider())
CreateCalorieData = composio.tools.get(user_id=os.environ("USER_ID"), tools=["NOTION_INSERT_ROW_DATABASE"])
ReadCalorieData = composio.tools.get(user_id=os.environ("USER_ID"), tools=["NOTION_QUERY_DATABASE"])

@tool
def SearchFoodCalorieAPI(food_name: str) -> dict:
    """
    Search for calorie information of various food items from an external database.

    Args:
        food_name (str): The name of the food item to search for.

    Returns:
        dict: A dictionary containing product information, including nutriments.
    """
    api = OpenFoodFactsAPI()
    products = api.search(food_name)
    if products:
        return products[0]  # Return the first product found
    return {}

@tool
def SearchExerciseCalorieAPI(activity: str) -> dict:
    """
    Search for calorie expenditure data for different types of exercises and durations.

    Args:
        activity (str): The name of the exercise activity.

    Returns:
        dict: A dictionary containing calorie expenditure data.
    """
    api_url = f'https://api.api-ninjas.com/v1/caloriesburned?activity={activity}'
    headers = {'X-Api-Key': os.environ.get('NINJA_API_KEY')}  # Assuming API key is in environment variable
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Error: {response.status_code} - {response.text}"}
'''