import yaml
from models.agent_architecture import AgentComponent, AgentRelation, AgentArchitecture
from langgraph_gen.generate import generate_from_spec
def convert_to_yaml(agent_architecture: AgentArchitecture):
    yaml_data = {
        "name": "AgenticRag",
        "entrypoint": "agent",
        "nodes": [{"name": component.name} for component in agent_architecture.agentComponents],
        "edges": [
            {
                "from": relation.source,
                "to": relation.target,
                "condition": getattr(relation, "relation_type", None),
                "paths": [relation.target] if relation.relation_type == "is_relevant" else None,
            }
            for relation in agent_architecture.agentRelations
        ],
    }

    # Removing None values from edges
    for edge in yaml_data["edges"]:
        edge.pop("condition", None) if edge.get("condition") is None else None
        edge.pop("paths", None) if edge.get("paths") is None else None

    return yaml.dump(yaml_data, default_flow_style=False)


if __name__ == "main":
    # Example usage
    agent_components = [
    AgentComponent(name="agent", description="The agent node"),
    AgentComponent(name="retrieve", description="The retrieval node"),
    AgentComponent(name="rewrite", description="The rewrite node"),
    AgentComponent(name="generate", description="The generation node"),
    ]
    agent_relations = [
        AgentRelation(source="agent", target="retrieve", relation_type=""),
        AgentRelation(source="retrieve", target="rewrite", relation_type="is_relevant"),
        AgentRelation(source="retrieve", target="generate", relation_type="is_relevant"),
        AgentRelation(source="rewrite", target="agent", relation_type=""),
        AgentRelation(source="generate", target="__end__", relation_type=""),
    ]

    agent_architecture = AgentArchitecture(agentComponents=agent_components, agentRelations=agent_relations)
    # Convert to YAML
    yaml_output = convert_to_yaml(agent_architecture)
    [stub, impl]= generate_from_spec(yaml_output, "yaml", ["stub", "implementation"], language="python")
    print("Stub Code:\n", stub)
    print("Implementation Code:\n", impl)
