"""This is an automatically generated file. Do not modify it.

This file was generated using `langgraph-gen` version 0.0.6.
To regenerate this file, run `langgraph-gen` with the source `yaml` file as an argument.

Usage:

1. Add the generated file to your project.
2. Create a new agent using the stub.

Below is a sample implementation of the generated stub:

```python
from typing_extensions import TypedDict

# Update the import path
# from [path to your stub] import AgenticRag

class SomeState(TypedDict):
    # define your attributes here
    foo: str

# Define stand-alone functions
def GoalTracker(state: SomeState) -> dict:
    print("In node: GoalTracker")
    return {
        # Add your state update logic here
    }


def MealLogger(state: SomeState) -> dict:
    print("In node: MealLogger")
    return {
        # Add your state update logic here
    }


def RecipeSuggester(state: SomeState) -> dict:
    print("In node: RecipeSuggester")
    return {
        # Add your state update logic here
    }


def GuidanceProvider(state: SomeState) -> dict:
    print("In node: GuidanceProvider")
    return {
        # Add your state update logic here
    }


def updates(state: SomeState) -> str:
    print("In condition: updates")
    raise NotImplementedError("Implement me.")


def requests(state: SomeState) -> str:
    print("In condition: requests")
    raise NotImplementedError("Implement me.")


def requests(state: SomeState) -> str:
    print("In condition: requests")
    raise NotImplementedError("Implement me.")


def updates(state: SomeState) -> str:
    print("In condition: updates")
    raise NotImplementedError("Implement me.")


agent = AgenticRag(
    state_schema=SomeState,
    impl=[
        ("GoalTracker", GoalTracker),
        ("MealLogger", MealLogger),
        ("RecipeSuggester", RecipeSuggester),
        ("GuidanceProvider", GuidanceProvider),
        ("updates", updates),
        ("requests", requests),
        ("requests", requests),
        ("updates", updates),
    ]
)

compiled_agent = agent.compile()

print(compiled_agent.invoke({"foo": "bar"}))
"""

from typing import Callable, Any, Optional, Type

from langgraph.constants import START, END  # noqa: F401
from langgraph.graph import StateGraph


def AgenticRag(
    *,
    state_schema: Optional[Type[Any]] = None,
    config_schema: Optional[Type[Any]] = None,
    input: Optional[Type[Any]] = None,
    output: Optional[Type[Any]] = None,
    impl: list[tuple[str, Callable]],
) -> StateGraph:
    """Create the state graph for AgenticRag."""
    # Declare the state graph
    builder = StateGraph(
        state_schema, config_schema=config_schema, input=input, output=output
    )

    nodes_by_name = {name: imp for name, imp in impl}

    all_names = set(nodes_by_name)

    expected_implementations = {
        "GoalTracker",
        "MealLogger",
        "RecipeSuggester",
        "GuidanceProvider",
        "updates",
        "requests",
        "requests",
        "updates",
    }

    missing_nodes = expected_implementations - all_names
    if missing_nodes:
        raise ValueError(f"Missing implementations for: {missing_nodes}")

    extra_nodes = all_names - expected_implementations

    if extra_nodes:
        raise ValueError(
            f"Extra implementations for: {extra_nodes}. Please regenerate the stub."
        )

    # Add nodes
    builder.add_node("GoalTracker", nodes_by_name["GoalTracker"])
    builder.add_node("MealLogger", nodes_by_name["MealLogger"])
    builder.add_node("RecipeSuggester", nodes_by_name["RecipeSuggester"])
    builder.add_node("GuidanceProvider", nodes_by_name["GuidanceProvider"])

    # Add edges
    builder.add_conditional_edges(
        "GoalTracker",
        nodes_by_name["updates"],
        [
        ],
    )
    builder.add_conditional_edges(
        "MealLogger",
        nodes_by_name["requests"],
        [
        ],
    )
    builder.add_conditional_edges(
        "MealLogger",
        nodes_by_name["requests"],
        [
        ],
    )
    builder.add_conditional_edges(
        "GuidanceProvider",
        nodes_by_name["updates"],
        [
        ],
    )
    builder.set_entry_point("agent")
    return builder
