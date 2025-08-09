from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command
from typing import Literal
from final_code.llms.model_factory import get_model
from final_code.utils.dependency_setup import get_embeddings, get_vector_store
from final_code.nodes_sup_style.node_to_code.node_to_code_base import NodeBuilderState

embeddings = get_embeddings()
vector_store = get_vector_store(embeddings, "langstuffindex")
llm = get_model()


prompt_vector_store = get_vector_store(embeddings, "promptguide")
prompt_retriever = prompt_vector_store.as_retriever()
prompt_gen_retriever = create_retriever_tool(prompt_retriever, "Retrieve_info_on_prompting", "Search information about what are different prompting techniques relevant to the user requirements.")

promptgen_prompt = ChatPromptTemplate.from_messages([
         ("system", """
    You are a ReAct (Reasoning and Act) agent.
    You are tasked with generating a prompt to meet the objectives of the langgraph node.
    The langgraph node information is provided. 
    
    For example: 
    User query: the node is supposed to generate a plan using llms
    Thought: I need to generate a prompt that will make the LLM generate a plan for the given task.
    Action: Use the  'Retrieve_info_on_prompting' tool to search for plan-and-execute prompting techniques.
    Observation: I found a plan-and-execute prompting technique that can generate a plan for the given task.
    Action: I will customize the observed prompt to meet the requirements of the node.
    
    IMPORTANT: Your final output will be only a prompt, no code
    """),
    ("placeholder", "{messages}"),
    ])
prompt_gen_agent = create_react_agent(llm, tools=[prompt_gen_retriever], prompt = promptgen_prompt)

def prompt_generation(state: NodeBuilderState) -> Command[Literal["ai_node_gen_supervisor"]]:
    """Generate the code for the node."""
    response = prompt_gen_agent.invoke({"messages": state["messages"]})
    return Command(
        update={
            "past_steps": [(state["task"], response["messages"][-1].content)],
            "messages": [
                HumanMessage(content=response["messages"][-1].content, name="prompt_generator")
            ]
        },
        goto="replan",
    )
