from e2b_code_interpreter import Sandbox, AsyncCommandHandle, AsyncSandbox
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
    start_time = time.time()
    sandbox = await AsyncSandbox.create(template='agrlcfpcd4qxe7ly7xro',timeout=60)
    await sandbox.files.write("/home/user/app.py", sample_langgraph_code)
    end_time = time.time()
    print(f"Time taken to initialize sandbox and write files: {end_time - start_time:.2f} seconds")
    
    start_time = time.time()
    process = await sandbox.commands.run("langgraph dev", cwd="/home/user", background=True, on_stdout=lambda data: print(data), on_stderr=lambda data: print(data))
    host = sandbox.get_host(2024)
    url = f"https://{host}"
    end_time = time.time()
    print(f'Server started at: {url} (took {end_time - start_time:.2f} seconds)')

    start_time = time.time()
    while time.time() - start_time < 30:
        try:
            response = requests.get(url + "/docs")
            if response.status_code == 200:
                data = response.text
                print(f"Response from server inside sandbox (took {time.time() - start_time:.2f} seconds):", data)
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1.5)

    
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


    # Kill the server process inside the sandbox.
    await process.kill()

if __name__ == "__main__":
    asyncio.run(main())
