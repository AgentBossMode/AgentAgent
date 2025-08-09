from langgraph.graph import START, END, StateGraph, MessagesState
import boto3
import json
import os
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional


app = FastAPI()


def node_a(state: MessagesState):
    return {"messages": ["node a"]}

def node_b(state: MessagesState):
    return {"messages": ["node b"]}

def node_c(state: MessagesState):
    return {"messages": ["node c"]}


workflow = StateGraph(MessagesState)
workflow.add_node("node_a", node_a)
workflow.add_node("node_b", node_b)
workflow.add_node("node_c", node_c)

workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", "node_c")
workflow.add_edge("node_c", END)

app = workflow.compile()

class Story(BaseModel):
    topic: Optional[str] = None


@app.post("/api/story")
def api_story(story: Story):
    if story.topic == None or story.topic == "":
        return None

    return StreamingResponse(bedrock_stream(story.topic))



async def bedrock_stream(topic: str):
    instruction = f"""
    You are a world class writer. Please write a sweet bedtime story about {topic}.
    """
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": instruction,
            }
        ],
    })

    stream = app.stream({}, stream_mode="updates")
    if stream:
        for event in stream:
            yield event


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
