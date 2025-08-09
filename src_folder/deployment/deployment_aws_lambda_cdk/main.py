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
import pickle

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

final_app = workflow.compile()

class Story(BaseModel):
    topic: Optional[str] = None

async def bedrock_stream(stream_instance):
    for event in stream_instance:
        print(event)
        yield pickle.dumps(event)

@app.get("/api/story")
def api_story(story: Story):
    if story.topic == None or story.topic == "":
        return None
    stream_instance = final_app.stream({}, stream_mode="values") 
    return StreamingResponse(bedrock_stream(stream_instance))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
