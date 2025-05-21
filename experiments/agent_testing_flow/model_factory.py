from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
from enum import Enum
from dotenv import load_dotenv
load_dotenv()

class ModelName(Enum):
    GPT4oMINI = 1
    GPT41MINI = 2
    CLAUDE37SONNET = 3
    GEMINI25FLASH = 4



model_dict = {
    ModelName.GPT4oMINI : ChatOpenAI(model="gpt-4o-mini", temperature=0),
    ModelName.GPT41MINI: ChatOpenAI(model="gpt-4.1-mini", temperature=0),
    ModelName.CLAUDE37SONNET: ChatAnthropic(model='claude-3-7-sonnet-20250219', temperature=0),
    ModelName.GEMINI25FLASH: ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0)
}

def get_model(model_name : ModelName= ModelName.GPT41MINI) -> BaseChatModel:
    """
    Retrieves a specific chat model instance from the model dictionary.

    Returns:
        BaseChatModel: The chat model instance corresponding to the key
    """
    return model_dict[model_name]