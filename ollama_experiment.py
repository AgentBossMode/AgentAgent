from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import asyncio
from dotenv import load_dotenv

load_dotenv()

model = ChatOllama(model="qwen2.5-coder:1.5b")

prompt = ChatPromptTemplate.from_template("Generate langgraph code")
parser = StrOutputParser()
chain = prompt | model | parser


async def main():
    async for event in chain.astream_events({}):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="|", flush=True)

asyncio.run(main())