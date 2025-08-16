from langgraph.types import Command
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from final_code.llms.model_factory import get_model
from final_code.nodes_sup_style.node_to_code.node_to_code_base import NodeBuilderState


llm = get_model()

interrupt_gen_prompt: str = """
    User will provide you with the information about the node, you are supposed to analyze the information and see if it requires interrupt functionality.
    
    Following is the information about the interrupt functionality, what is the purpose of interrupt, design patterns, how to implement them:
    <InterruptInfo>
    The interrupt function in LangGraph enables human-in-the-loop workflows by pausing the graph at a specific node, presenting information to a human, and resuming the graph with their input. This function is useful for tasks like approvals, edits, or collecting additional input. The interrupt function is used in conjunction with the Command object to resume the graph with a value provided by the human.


``` python
from langgraph.types import interrupt

def human_node(state: State):
    value = interrupt(
        # Any JSON serializable value to surface to the human.
        # For example, a question or a piece of text or a set of keys in the state
       {
          "text_to_revise": state["some_text"]
       }
    )
    # Update the state with the human's input or route the graph based on the input.
    return {
        "some_text": value
    }

graph = graph_builder.compile(
    checkpointer=checkpointer # Required for `interrupt` to work
)

# Run the graph until the interrupt
thread_config = {"configurable": {"thread_id": "some_id"}}
graph.invoke(some_input, config=thread_config)

# Resume the graph with the human's input
graph.invoke(Command(resume=value_from_human), config=thread_config)
```
    </InterruptInfo>
    
    <Output>
    Unless explicitly mentioned in node requirements, human-in-loop aka interrupt is not needed in the scenario.
    If needed: implement the code with interrupt functionality tailored to the use case
    If not needed: just respond that interrupt functionality is not needed
    </Output>
    """

def interrupt_generation(state: NodeBuilderState) -> Command[Literal["ai_node_gen_supervisor"]]:
    """Generate the code for the node."""
    # fetch_docs=fetch_documents("https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/")
    response = llm.invoke( [SystemMessage(content=interrupt_gen_prompt)]  + state["messages"])
    return Command(
        update={
            "past_steps": [(state["task"], response.content)],
            "messages": [
                HumanMessage(content=response.content, name="interrupt_generator")
            ]
        },
        goto="replan",
    )
