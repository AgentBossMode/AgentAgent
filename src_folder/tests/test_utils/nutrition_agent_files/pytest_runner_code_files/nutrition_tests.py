nutrition_tests = r'''

import pytest
from app import app
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI


from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
import pytest
from uuid import uuid4


# LLM-as-judge instructions
grader_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION, a REFERENCE RESPONSE, and the STUDENT RESPONSE.

You are grading whether the student's response is appropriate to the question and matches the reference response in spirit.

Correctness:
True means that the student's response meets the criteria.
False means that the student's response does not meet the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct."""

# LLM-as-judge output schema
class Grade(BaseModel):
    """Compare the expected and actual answers and grade the actual answer."""
    reasoning: str = Field(description="Explain your reasoning for whether the actual response is correct or not.")
    is_correct: bool = Field(description="True if the student response is mostly or exactly correct, otherwise False.")

# Judge LLM
grader_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Grade)

def final_answer_correct(input: str, reference_output: str, actual_output: str) -> bool:
    """Evaluate if the final response is equivalent to reference response."""

    # Note that we assume the outputs has a 'response' dictionary. We'll need to make sure
    # that the target function we define includes this key.
    user = f"""QUESTION: {input}
    GROUND TRUTH RESPONSE: {reference_output}
    STUDENT RESPONSE: {actual_output}"""

    grade: Grade = grader_llm.invoke([{"role": "system", "content": grader_instructions}, {"role": "user", "content": user}])
    return grade.is_correct

@pytest.mark.parametrize(
    "input_query, reference_output",
    [
        ("I ate a banana and ran for 30 minutes.", "Understood. A banana is approximately 105 calories. A 30-minute run typically burns around 300-450 calories, depending on intensity. This data has been recorded."),
("I ran for 20 minutes.", "Okay, a 20-minute run has been logged. This typically burns around 200-300 calories. This data has been stored."),
("How many calories did I burn yesterday?", "Yesterday, you burned a total of [X] calories through your activities."),
("What did I eat on Monday?", "On Monday, you reported eating: [List of food items and their calorie counts]."),
("What's my net calorie intake for the week?", "This week, your total calorie intake was [X] and your total calories burned were [Y], resulting in a net of [X-Y] calories.")
    ],
)
def test_full_workflow_final_response(input_query: str, reference_output: str):
    # Invoke the full graph
    thread_config = {"configurable": {"thread_id": uuid4() }}
    result = app.invoke({"messages": [HumanMessage(content=input_query)]}, config=thread_config)

    # Get the last message, which is the final response
    actual_output = result["messages"][-1].content
    assert final_answer_correct(input, reference_output, actual_output) == True
    



# LLM-as-judge instructions
grader_trajectory_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION, a REFERENCE RESPONSE, and the STUDENT RESPONSE.

You are grading whether the student's response is appropriate to the question and matches the reference response in spirit.

<OUTPUT_INSTRUCTIONS>
is_correct:
True means that the student's response meets the criteria.
False means that the student's response does not meet the criteria.

reasoning should follow the following format:
STUDENT TRAJECTORY: list of student trajectory displayed
GROUND TRUTH TRAJECTORY: list of ground truth trajectory displayed
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
</OUTPUT_INSTRUCTIONS>

"""



# LLM-as-judge output schema
class GradeTrajectory(BaseModel):
    """Compare the expected and actual answers and grade the actual answer."""
    reasoning: str = Field(description="Explain your reasoning for whether the actual response is correct or not.")
    is_correct: bool = Field(description="True if the student response is mostly or exactly correct, otherwise False.")

# Judge LLM
grade_trajectory_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(GradeTrajectory)

@pytest.mark.parametrize(
    "input_query, expected_tool_call_names",
    [
     ("I ate a banana and ran for 30 minutes.", ['calorie_calculation', 'data_storage']),
("I ran for 20 minutes.", ['calorie_calculation', 'data_storage']),
("How many calories did I burn yesterday?", ['calorie_calculation', 'historical_retrieval']),
("What did I eat on Monday?", ['calorie_calculation', 'historical_retrieval']),
("What's my net calorie intake for the week?", ['calorie_calculation', 'net_calorie_analysis'])
    ]
)
def test_full_workflow_trajectory(input_query: str, expected_tool_call_names: list[str]):   
    trajectory = []
    thread_config = {"configurable": {"thread_id": uuid4()}}

    for namespace, chunk in app.stream({"messages": [
            {
                "role": "user",
                "content": input_query,
            }]
            }, config=thread_config, subgraphs=True, stream_mode="debug"):
        # Event type for entering a node
        if chunk['type'] == 'task':
            # Record the node name
            trajectory.append(chunk['payload']['name'])
            # Given how we defined our dataset, we also need to track when specific tools are
            # called by our question answering ReACT agent. These tool calls can be found
            # when the ToolsNode (named "tools") is invoked by looking at the AIMessage.tool_calls
            # of the latest input message.
            # if chunk['payload']['name'] == 'tools' and chunk['type'] == 'task':
            #    print(chunk['payload']['input']['messages'][-1])
            #    for tc in chunk['payload']['input']['messages'][-1].tool_calls:
            #        trajectory.append(tc['name'])
    grading_assignment = f"""QUESTION: {input_query}
    GROUND TRUTH TRAJECTORY: {" ".join(expected_tool_call_names)}
    STUDENT TRAJECTORY: {" ".join(trajectory)}"""
    grade: GradeTrajectory = grade_trajectory_llm.invoke([{"role": "system", "content": grader_trajectory_instructions}, {"role": "user", "content": grading_assignment}])
    return grade.is_correct

'''