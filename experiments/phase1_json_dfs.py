import import_ipynb
from experiments.edge_to_code.edge_to_code import edge_builder_agent
from experiments.node_to_code.node_to_code import node_to_code_app
import operator
from typing import Annotated, List
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Send
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from experiments.model_factory import get_model, ModelName
import uuid
import json

skipList = ["__START__", "__END__"]

class NodeEvaluationReport(BaseModel):
    node_name: str
    node_code_stub: str
    edge_code: str

class GraphCompilerState(MessagesState):
    json_code: str
    node_reports: Annotated[List[NodeEvaluationReport], operator.add]


class NodeProcessState():
    node_name: str
    node_info: dict
    edge_info: dict

def graph_map_step(state: GraphCompilerState):
    # Extract nodes and edges from json_objects
    json_objects = json.loads(state["json_code"])
    nodes = json_objects["nodes"]
    edges = json_objects["edges"]
    
    sends = []
    
    for node in nodes:
        outgoing_edges = [edge for edge in edges if edge["source"] == node["id"]]
        sends.append(Send("node_process", {"node_name": node["id"], "node_info": node, "edge_info": outgoing_edges}))    
    return sends

edge_info_prompt = ChatPromptTemplate.from_template("""
<GraphNodeImplementation>
{node_code}
</GraphNodeImplementation>
<EdgeInformation>
{edge_json}
</Edgeinformation>""")

def node_process(state: NodeProcessState):
    uuid_str = uuid.uuid4()
    config = {"configurable": {"thread_id": str(uuid_str)}}
    if state["node_name"] not in skipList:
        for output in node_to_code_app.stream(state["node_info"], config, stream_mode="updates"):
            print(output)
        code= node_to_code_app.get_state(config).values["final_code"]
    else:
        code = "no implementation needed"
    edge_code = edge_builder_agent.invoke({"messages": [HumanMessage(content = edge_info_prompt.format(node_code=code,edge_json=state["edge_info"]))]},config)
    return {
        "node_reports" : [NodeEvaluationReport(node_name=state["node_name"], node_code_stub=code, edge_code=edge_code["messages"][-1].content)] 
    }

code_compiler_prompt = ChatPromptTemplate.from_template("""
<Graph>
{graphEdgeDict}
</Graph>

<GraphNodeImplementation>
{graphImplementations}
</GraphNodeImplementation>

<EdgeImplementation>
{edgeImplementations}
</EdgeImplementation>
""")

def graph_compile(state: GraphCompilerState):
    node_evals : List[NodeEvaluationReport]= state["node_reports"]

    code_stubs = [node_eval.node_code_stub for node_eval in node_evals]
    edge_stubs = [node_eval.edge_code for node_eval in node_evals]
    json_objects = json.loads(state["json_code"])
    edges = json_objects["edges"]
    llm = get_model()
    response = llm.invoke([SystemMessage(content=
"""You are a langgraph coding expert, you are given a workflow with edges as well as code implementation of each node.
You are supposed to merge the code, make sure there is no simulation of llm operations, use langchain for llm invocations
Make sure that the graph is compiled with an InMemoryCheckpointer and finally assign to a variable called final_app""")]
                          +[HumanMessage(content= code_compiler_prompt.format(
        graphEdgeDict=edges,
        graphImplementations=code_stubs,
        edgeImplementations=edge_stubs))])
    return {
        "messages": [response]
    }


workflow = StateGraph(GraphCompilerState)
workflow.add_node(graph_map_step, "graph_map")
workflow.add_node(node_process, "node_process")
workflow.add_node(graph_compile,"graph_compile")

workflow.add_conditional_edges(START, graph_map_step, ["node_process"])
workflow.add_edge("node_process", "graph_compile")
workflow.add_edge("graph_compile", END)

compiler_graph = workflow.compile()