from pydantic import BaseModel, Field
class EndOrContinue(BaseModel):
    should_end_conversation : bool = Field(description="true if the AI response does not indicate that it needs any human input.")
