from langgraph_sdk import get_client
from langchain_core.messages import HumanMessage
import asyncio


url="http://localhost:8123"
client = get_client(url=url)
assistant_id = "app"

async def main():
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


if __name__ == "__main__":
    asyncio.run(main())