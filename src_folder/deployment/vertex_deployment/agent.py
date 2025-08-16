from typing import Any, Mapping, Optional
from vertexai import agent_engines
from langchain_core.language_models import BaseLanguageModel
import os
import vertexai
from vertexai import agent_engines
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv

load_dotenv()

# vertexai.init(
#     project="cloud-run-testing-460718",               # Your project ID.
#     location="us-central1",                # Your cloud region.
#     staging_bucket="gs://bucket1kanishk",  # Your staging bucket.
# )

DB_URI=os.environ["DB_URI"]
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    def langgraph_builder(*, model, **kwargs):
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langgraph.graph import END, MessageGraph

        output_parser = StrOutputParser()

        planner = ChatPromptTemplate.from_template(
            "Generate an argument about: {input}"
        ) | model | output_parser

        pros = ChatPromptTemplate.from_template(
            "List the positive aspects of {input}"
        ) | model | output_parser

        cons = ChatPromptTemplate.from_template(
            "List the negative aspects of {input}"
        ) | model | output_parser

        summary = ChatPromptTemplate.from_template(
            "Input:{input}\nGenerate a final response given the critique",
        ) | model | output_parser

        builder = MessageGraph()
        builder.add_node("planner", planner)
        builder.add_node("pros", pros)
        builder.add_node("cons", cons)
        builder.add_node("summary", summary)

        builder.add_edge("planner", "pros")
        builder.add_edge("planner", "cons")
        builder.add_edge("pros", "summary")
        builder.add_edge("cons", "summary")
        builder.add_edge("summary", END)
        builder.set_entry_point("planner")
        return builder.compile(checkpointer=checkpointer)

    def openAI_model_Builder(
        model_name: str,
        *,
        project: str,
        location: str,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> "BaseLanguageModel":
        """Default callable for building a language model.

        Args:
            model_name (str):
                Required. The name of the model (e.g. "gemini-1.0-pro").
            project (str):
                Required. The Google Cloud project ID.
            location (str):
                Required. The Google Cloud location.
            model_kwargs (Mapping[str, Any]):
                Optional. Additional keyword arguments for the constructor of
                chat_models.ChatVertexAI.

        Returns:
            BaseLanguageModel: The language model.
        """
        from langchain_openai import ChatOpenAI

        model_kwargs = model_kwargs or {}
        model = ChatOpenAI(model_name=model_name, temperature=0)
        return model


    agent = agent_engines.LanggraphAgent(
        model="gpt-4o-mini",
        runnable_builder=langgraph_builder,
        model_builder=openAI_model_Builder
    )

    for state_updates in agent.stream_query(
        input={"role": "user", "content": "scrum methodology"},
        stream_mode="updates",
        config={"configurable": {"thread_id": "streaming-thread-updates"}}):
        print(state_updates)


# if __name__ == "__main__":
#     print("hello")
#     remote_agent = agent_engines.create(
#         agent,                    # Optional.
#         requirements = [
#         "google-cloud-aiplatform[agent_engines,langgraph]",
#         "langgraph",
#         "langchain-openai"],
#         env_vars={
#             "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
#         },              # Optional.
#     )

#     # Example query
#     # response = agent.query(input=)

#     for state_updates in agent.stream_query(
#         input={"role": "user", "content": "scrum methodology"},
#         stream_mode="updates",
#         config={"configurable": {"thread_id": "streaming-thread-updates"}}):
#         print(state_updates)

#     # print(response)
#     remote_agent.delete(force=True)