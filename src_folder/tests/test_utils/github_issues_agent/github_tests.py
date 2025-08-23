github_tests =r'''
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

def final_answer_correct(input_query: str, reference_output: str, actual_output: str) -> bool:
    """Evaluate if the final response is equivalent to reference response."""

    # Note that we assume the outputs has a 'response' dictionary. We'll need to make sure
    # that the target function we define includes this key.
    user = f"""QUESTION: {input_query}
    GROUND TRUTH RESPONSE: {reference_output}
    STUDENT RESPONSE: {actual_output}"""

    grade: Grade = grader_llm.invoke([{"role": "system", "content": grader_instructions}, {"role": "user", "content": user}])
    return grade.is_correct

# @pytest.mark.parametrize(
#     "input_query, reference_output",
#     [
#         ("Email from user: 'Subject: Website Login Broken. Body: I cannot log in to the website. It keeps showing an error message after I enter my credentials.'", "Email sent to user: 'Your reported issue regarding website login is already being tracked under #xfef. We are working on it.'"),
# ("Email from user: 'Subject: UI Glitch on Homepage. Body: The navigation bar on the homepage is overlapping with the main content on mobile devices.'", "New issue created in 'PromptiusWeb' repository. Email sent to user: 'Thank you for reporting the UI glitch on the homepage. A new issue has been created and is being tracked under #new_issue_id.'"),
# ("Email from user: 'Subject: Agent Response Delay. Body: The agent is taking too long to respond to my queries, sometimes over 5 minutes.'", "New issue created in 'AgentAgent' repository. Email sent to user: 'Thank you for reporting the agent response delay. A new issue has been created and is being tracked under #new_agent_issue_id.'"),
# ("Email from user: 'Subject: Billing Inquiry. Body: I was charged twice for my last subscription renewal. Please investigate.'", "Support ticket created in the Ticketing System. Email sent to user: 'Thank you for your billing inquiry. A support ticket has been created and is being tracked under #ticket_number. Our team will investigate and get back to you shortly.'")
#     ],
# )
# def test_full_workflow_final_response(input_query: str, reference_output: str):
#     # Invoke the full graph
#     thread_config = {"configurable": {"thread_id": uuid4() }}
#     result = app.invoke({"messages": [HumanMessage(content=input_query)]}, config=thread_config)

#     # Get the last message, which is the final response
#     actual_output = result["messages"][-1].content.strip()
#     assert final_answer_correct(input_query, reference_output, actual_output) == True
    

from agentevals.graph_trajectory.llm import create_graph_trajectory_llm_as_judge
from agentevals.graph_trajectory.utils import (
    extract_langgraph_trajectory_from_thread,
)

DEFAULT_REF_COMPARE_PROMPT = """You are an expert data labeler.
Your task is to grade the accuracy of an AI agent's internal steps in resolving a user queries.

<Rubric>
  An accurate trajectory:
  - Makes logical sense between steps
  - Shows clear progression
  - Is relatively efficient, though it does not need to be perfectly efficient
  - Is semantically equivalent to the provided reference trajectory, if present
</Rubric>

<Instructions>
  Grade the following thread, evaluating whether the agent's overall steps are logical and relatively efficient.
  Check the list of "messages" passed, read each of them and see if the progression of ai messages is logical.
  For the trajectory, "__start__" denotes an initial entrypoint to the agent, and "__interrupt__" corresponds to the agent
  interrupting to await additional data from another source ("human-in-the-loop"):
</Instructions>

<thread>
{thread}
</thread>

{reference_outputs}
"""

def serialize_pydantic_models(data):
    from pydantic import BaseModel
    from langchain_core.messages import BaseMessage
    if isinstance(data, dict):
        return {{k: serialize_pydantic_models(v) for k, v in data.items()}}
    elif isinstance(data, list):
        return [serialize_pydantic_models(i) for i in data]
    elif isinstance(data, BaseModel) and not isinstance(data, BaseMessage):
        return data.model_dump()
    return data

@pytest.mark.parametrize(
    "input_query, expected_tool_call_names",
    [
     ("Email from user: 'Subject: Website Login Broken. Body: I cannot log in to the website. It keeps showing an error message after I enter my credentials.'", ['read_email', 'analyze_email', 'search_github_issues', 'respond_to_user']),
("Email from user: 'Subject: UI Glitch on Homepage. Body: The navigation bar on the homepage is overlapping with the main content on mobile devices.'", ['read_email', 'analyze_email', 'search_github_issues', 'create_github_issue_website', 'respond_to_user']),
("Email from user: 'Subject: Agent Response Delay. Body: The agent is taking too long to respond to my queries, sometimes over 5 minutes.'", ['read_email', 'analyze_email', 'search_github_issues', 'create_github_issue_agent', 'respond_to_user']),
("Email from user: 'Subject: Billing Inquiry. Body: I was charged twice for my last subscription renewal. Please investigate.'", ['read_email', 'analyze_email', 'raise_support_ticket', 'respond_to_user'])
    ]
)
def test_full_workflow_trajectory(input_query: str, expected_tool_call_names: list[str]):   
    thread_config = {{"configurable": {{"thread_id": uuid4()}}}}
    result = app.invoke({{"messages": [
            {{
                "role": "user",
                "content": input_query,
            }}]
            }}, config=thread_config)
    
    # Extract the trajectory from the first two thread runs
    extracted_trajectory = extract_langgraph_trajectory_from_thread(app, thread_config)
    
    serializable_result = serialize_pydantic_models(result)
    extracted_trajectory["outputs"]["results"]= [serializable_result]
    graph_trajectory_evaluator = create_graph_trajectory_llm_as_judge(prompt=DEFAULT_REF_COMPARE_PROMPT, model="openai:gpt-4o-mini")
    print(extracted_trajectory)
    res = graph_trajectory_evaluator(inputs=extracted_trajectory["inputs"], outputs=extracted_trajectory["outputs"])
    assert res["score"] == True, res["comment"]
'''