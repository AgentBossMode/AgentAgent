def write_trajectory_pytest_code(query: list[str], trajectory: list[list[str]]) -> str:
    """
    Generate pytest code for testing the trajectory of tool calls in a workflow.

    Args:
        query (list[str]): List of input queries to test against the workflow
        trajectory (list[list[str]]): List of expected tool call trajectories for each query

    Returns:
        str: Formatted pytest code that tests if the actual tool call trajectory matches the expected trajectory
    """
    result = ""
    for i, (q, t) in enumerate(zip(query, trajectory)):
        if i == len(query) - 1:
            result += f'("{q}", {t})'
        else:
            result += f'("{q}", {t}),\n'
    code_to_format ="""

# LLM-as-judge instructions
grader_instructions = \"\"\"You are a teacher grading a quiz.

You will be given a QUESTION, a REFERENCE RESPONSE, and the STUDENT RESPONSE.

You are grading whether the student's response is appropriate to the question and matches the reference response in spirit.

Correctness:
True means that the student's response meets the criteria.
False means that the student's response does not meet the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.\"\"\"

# LLM-as-judge output schema
class GradeTrajectory(BaseModel):
    \"\"\"Compare the expected and actual answers and grade the actual answer.\"\"\"
    reasoning: str = Field(description="Explain your reasoning for whether the actual response is correct or not.")
    is_correct: bool = Field(description="True if the student response is mostly or exactly correct, otherwise False.")

# Judge LLM
grade_trajectory_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(GradeTrajectory)

@pytest.mark.parametrize(
    "input_query, expected_tool_call_names",
    [
     {result}
    ]
)
def test_full_workflow_trajectory(input_query: str, expected_tool_call_names: list[str]):   
    trajectory = []
    for namespace, chunk in app.stream({{"messages": [
            {{
                "role": "user",
                "content": input_query,
            }}]
            }}, subgraphs=True, stream_mode="debug"):
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
    grading_assignment = f\"\"\"QUESTION: {{input_query}}
    GROUND TRUTH RESPONSE: {{" ".join(expected_tool_call_names)}}
    STUDENT RESPONSE: {{" ".join(trajectory}}\"\"\"
    grade: GradeTrajectory = grade_trajectory_llm.invoke([{{"role": "system", "content": grader_instructions}}, {{"role": "user", "content": grading_assignment}}])
    return grade.is_correct
"""
    return code_to_format.format(result=result)


def write_final_response_pytest_code(query: list[str], responses: list[str]) -> str:
    """Generate pytest code for testing final responses against expected outputs.
    
    Args:
        query (list[str]): List of input queries to test
        responses (list[str]): List of expected responses corresponding to each query
        
    Returns:
        str: Generated pytest code as a string containing test cases
    """
    result = ""
    for i, (q, t) in enumerate(zip(query, responses)):
        if i == len(query) - 1:
            result += f'("{q}", "{t}")'
        else:
            result += f'("{q}", "{t}"),\n'
    code="""
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
import pytest

# LLM-as-judge instructions
grader_instructions = \"\"\"You are a teacher grading a quiz.

You will be given a QUESTION, a REFERENCE RESPONSE, and the STUDENT RESPONSE.

You are grading whether the student's response is appropriate to the question and matches the reference response in spirit.

Correctness:
True means that the student's response meets the criteria.
False means that the student's response does not meet the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.\"\"\"

# LLM-as-judge output schema
class Grade(BaseModel):
    \"\"\"Compare the expected and actual answers and grade the actual answer.\"\"\"
    reasoning: str = Field(description="Explain your reasoning for whether the actual response is correct or not.")
    is_correct: bool = Field(description="True if the student response is mostly or exactly correct, otherwise False.")

# Judge LLM
grader_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Grade)

def final_answer_correct(input: str, reference_output: str, actual_output: str) -> bool:
    \"\"\"Evaluate if the final response is equivalent to reference response.\"\"\"

    # Note that we assume the outputs has a 'response' dictionary. We'll need to make sure
    # that the target function we define includes this key.
    user = f\"\"\"QUESTION: {{input}}
    GROUND TRUTH RESPONSE: {{reference_output}}
    STUDENT RESPONSE: {{actual_output}}\"\"\"

    grade: Grade = grader_llm.invoke([{{"role": "system", "content": grader_instructions}}, {{"role": "user", "content": user}}])
    return grade.is_correct

@pytest.mark.parametrize(
    "input_query, reference_output",
    [
        {result}
    ],
)
def test_full_workflow_final_response(input_query: str, reference_output: str):
    # Invoke the full graph
    result = app.invoke({{"messages": [HumanMessage(content=input_query)]}})

    # Get the last message, which is the final response
    actual_output = result["messages"][-1].content
    assert final_answer_correct(input, reference_output, actual_output) == True
    """
    return code.format(result=result)