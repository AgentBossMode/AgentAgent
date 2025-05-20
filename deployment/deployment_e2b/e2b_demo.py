from e2b_code_interpreter import Sandbox
import requests
import time
import os
from langgraph_sdk import get_client
from langchain_core.messages import HumanMessage
import asyncio

sample_langgraph_code="""
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage

def basic_node(state: MessagesState):
    return {"messages": [HumanMessage(content="Hello, how are you?")]}

workflow = StateGraph(MessagesState)
workflow.add_node("basic_node", basic_node)
workflow.add_edge(START, "basic_node")
workflow.add_edge("basic_node", END)

app = workflow.compile()
"""

sample_langgraph_json="""
{
    "graphs": {
      "app" : "./app.py:app"
    },
    "dependencies": [
      "./"
    ],
    "env": ".env"
}
"""



async def main():
    # add two files to the sandbox, first is langgraph.json and second is langgraph_demo.py
    sandbox = Sandbox(envs= {"OPENAI_API_KEY" : os.environ["OPENAI_API_KEY"]}, timeout=60)
    sandbox.commands.run("pip install langgraph-cli[inmem] langgraph langchain")
    sandbox.files.write("/home/user/app.py", sample_langgraph_code)
    sandbox.files.write("/home/user/langgraph.json", sample_langgraph_json)
    # Start a simple HTTP server inside the sandbox.
    process = sandbox.commands.run("langgraph dev", cwd="/home/user", background=True, on_stdout=lambda data: print(data), on_stderr=lambda data: print(data))
    host = sandbox.get_host(2024)
    url = f"https://{host}"
    print('Server started at:', url)

    time.sleep(10)
    client = get_client(url=url)
    assistant_id = "app"

    # create a thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    async for chunk in client.runs.stream(
        thread_id,
        assistant_id,
        input={"messages": [HumanMessage(content="Hello, how are you?")]},
        stream_mode="updates"
    ):
        print(chunk.data)
    # Fetch data from the server inside the sandbox.
    response = requests.get(url+ "/docs")
    data = response.text
    print("Response from server inside sandbox:", data)

    # Kill the server process inside the sandbox.
    process.kill()

if __name__ == "__main__":
    asyncio.run(main())
