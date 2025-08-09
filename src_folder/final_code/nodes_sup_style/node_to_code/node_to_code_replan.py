from pydantic import Field, BaseModel
from typing import List, Tuple, Union
from langchain_core.prompts import ChatPromptTemplate
from final_code.llms.model_factory import get_model
from final_code.nodes_sup_style.node_to_code.node_to_code_base import NodeBuilderState
from final_code.nodes_sup_style.node_to_code.node_to_code_planner import node_building_strategies, Plan

llm = get_model()

class Response(BaseModel):
    """Response to user."""
    response: str

class Act(BaseModel):
    """Action to perform."""
    justification: str = Field(description= "Step by step thinking to Justify choosing the action taken")

    action: Union[Response, Plan] = Field(
        description="Action to perform. If the original plan is complete, use Response, ele use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """
You are an expert code writer tasked with assessing a coding plan.
You have two actions to choose from: 'Response' or 'Plan'
You will be provided an 'OriginalPlan' which shows the list of planned steps you previously generated for implementing a function
You will be also provided 'PlanStepsExecuted' which shows a list of execution of a subset of the steps.

<OriginalPlan>
{plan}
</OriginalPlan>

<PlanStepsExecuted>
{past_steps}
</PlanStepsExecuted>

<NODEBUILDINGSTRATEGIES>
{node_building_strategies}
</NODEBUILDINGSTRATEGIES>


<INSTRUCTIONS>
1. If the <PlanStepsExecuted> section covers all the steps in the <OriginalPlan>, respond back to the user using 'Response'.
2. Your job is only to analyze if all the steps of <OriginalPlan> has been completed by using past_steps, do not suggest new steps or follow up steps.
3. Do not generate steps which are out of scope of the <OriginalPlan>, steps will always be in the scope of <NODEBUILDINGSTRATEGIES>
4. If there are steps that still need to be taken, respond with 'Plan' action. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.
</INSTRUCTIONS>
""")


replanner = replanner_prompt | llm.with_structured_output(Act)

def replan_step(state: NodeBuilderState):
    past_steps : List[Tuple] = state["past_steps"]
    past_steps_summary = "\n".join([f"{index}. <step>{step}</step> <output>{output}</output>" for index, (step, output) in enumerate(past_steps)])
    output = replanner.invoke({"plan": "\n".join(plan_step for plan_step in state["plan"]),
                               "past_steps": past_steps_summary,
                               "node_building_strategies": node_building_strategies})
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}
