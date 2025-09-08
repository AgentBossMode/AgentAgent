from pydantic import BaseModel, Field

class Questions(BaseModel):
        questions: list[str] = Field(description="List of questions to ask the user, explain why you are asking this question")
