from typing import Dict, Any, List, Optional, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src_folder.final_code.utils.create_react_agent_temp import create_react_agent
from langgraph.types import interrupt, Command
import re
import json
import bs4 
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

MOCK_TOOL_PROMPT = """
You are a helpful assistant that generates mock data for tool outputs.
Given the tool's purpose and expected output, generate a realistic mock response.
"""

INPUT_PROMPT = """
Tool Docstring: {description}
Input: {input}
Generate a mock output for this tool.
"""


def Web_Scraper_Parse_URL_and_Extract_Data(url: str):
    """
    A simple web scraper using BeautifulSoup4 to extract all text content from a given URL.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        str: All text content extracted from the webpage.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        return f"Error scraping URL {url}: {e}"

def Contact_Finder_Search_and_Retrieve_Contact(company_name: str, senior_executive_title: Optional[str] = None):
    """
    Tool to search for and retrieve contact information, specifically email addresses or LinkedIn profiles, of senior executives within a company.

    Args:
        company_name (str): The name of the company to search for contacts.
        senior_executive_title (Optional[str]): The title of the senior executive to search for (e.g., "CEO", "CTO").

    Returns:
        str: A JSON string containing the contact information.

    Example:
        Contact_Finder_Search_and_Retrieve_Contact(company_name="Google", senior_executive_title="CEO")
        # Expected output:
        # {
        #   "name": "Sundar Pichai",
        #   "email": "sundar.pichai@google.com",
        #   "linkedin_profile": "https://www.linkedin.com/in/sundarpichai"
        # }
    """
    class ContactInfo(BaseModel):
        name: str = Field(description="The name of the contact.")
        email: Optional[str] = Field(description="The email address of the contact.")
        linkedin_profile: Optional[str] = Field(description="The LinkedIn profile URL of the contact.")
        title: Optional[str] = Field(description="The job title of the contact.")

    input_str = f"company_name: {company_name}, senior_executive_title: {senior_executive_title}"
    description = Contact_Finder_Search_and_Retrieve_Contact.__doc__

    result = llm.with_structured_output(ContactInfo).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

def Email_Sender_Send_Email(to: str, subject: str, body: str, cc: Optional[List[str]] = None, bcc: Optional[List[str]] = None, attachments: Optional[List[str]] = None):
    """
    Tool to send a personalized email to the identified senior executive.

    Args:
        to (str): The recipient's email address.
        subject (str): The subject of the email.
        body (str): The main content of the email.
        cc (Optional[List[str]]): A list of email addresses to CC.
        bcc (Optional[List[str]]): A list of email addresses to BCC.
        attachments (Optional[List[str]]): A list of file paths for attachments.

    Returns:
        str: A JSON string indicating the email sending status.

    Example:
        Email_Sender_Send_Email(to="recipient@example.com", subject="Job Application", body="Dear Sir/Madam, ...")
        # Expected output:
        # {
        #   "status": "success",
        #   "message": "Email sent successfully."
        # }
    """
    class EmailStatus(BaseModel):
        status: str = Field(description="The status of the email sending operation (e.g., 'success', 'failed').")
        message: str = Field(description="A descriptive message about the email sending status.")

    input_str = f"to: {to}, subject: {subject}, body: {body}, cc: {cc}, bcc: {bcc}, attachments: {attachments}"
    description = Email_Sender_Send_Email.__doc__

    result = llm.with_structured_output(EmailStatus).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

def web_scrape_job_details(job_link: str):
    """
    Extracts job details (title, pay, duration, skills, company) from a given job link.

    Args:
        job_link (str): The URL of the job posting.

    Returns:
        str: A JSON string containing the extracted job details.

    Example:
        web_scrape_job_details(job_link="https://example.com/job/456")
        # Expected output:
        # {
        #   "job_link": "https://example.com/job/456",
        #   "job_details": {
        #     "title": "Data Scientist",
        #     "pay": "$100,000 - $130,000",
        #     "duration": "Full-time",
        #     "skills": ["Python", "R", "Machine Learning"],
        #     "company": "Innovate Corp"
        #   }
        # }
    """
    class JobDetailsOutput(BaseModel):
        title: str = Field(description="The job title.")
        pay: Optional[str] = Field(description="The pay or salary range for the job.")
        duration: Optional[str] = Field(description="The duration of the job (e.g., 'Full-time', 'Contract').")
        skills: List[str] = Field(description="A list of required skills for the job.")
        company: str = Field(description="The name of the company offering the job.")

    class GraphStateOutput(BaseModel):
        job_link: str = Field(description="The original job link.")
        job_details: JobDetailsOutput = Field(description="Extracted details of the job.")

    input_str = f"job_link: {job_link}"
    description = web_scrape_job_details.__doc__

    result = llm.with_structured_output(GraphStateOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

def find_contact_information(job_details: str):
    """
    Identifies and retrieves contact information for senior executives within a company based on extracted job details.

    Args:
        job_details (str): A JSON string containing the extracted job details, including the company name.

    Returns:
        str: A JSON string containing the contact information.

    Example:
        find_contact_information(job_details='''{"title": "Software Engineer", "company": "Tech Solutions Inc.", "pay": "$120,000 - $150,000 per year", "duration": "Full-time", "skills": ["Python", "Django", "AWS"]}''')
        # Expected output:
        # {
        #   "job_details": {
        #     "title": "Software Engineer",
        #     "company": "Tech Solutions Inc.",
        #     "pay": "$120,000 - $150,000 per year",
        #     "duration": "Full-time",
        #     "skills": ["Python", "Django", "AWS"]
        #   },
        #   "contact_info": {
        #     "name": "Jane Doe",
        #     "email": "jane.doe@techsolutions.com",
        #     "linkedin_profile": "https://www.linkedin.com/in/janedoe"
        #   }
        # }
    """
    class JobDetailsInput(BaseModel):
        title: str
        company: str
        pay: Optional[str]
        duration: Optional[str]
        skills: List[str]

    class ContactInfoOutput(BaseModel):
        name: str = Field(description="The name of the contact.")
        email: Optional[str] = Field(description="The email address of the contact.")
        linkedin_profile: Optional[str] = Field(description="The LinkedIn profile URL of the contact.")

    class GraphStateOutput(BaseModel):
        job_details: JobDetailsInput = Field(description="The original job details.")
        contact_info: ContactInfoOutput = Field(description="The retrieved contact information.")

    input_str = f"job_details: {job_details}"
    description = find_contact_information.__doc__

    result = llm.with_structured_output(GraphStateOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

def generate_email_content(job_details: str, contact_info: str, product_prd: str):
    """
    Generates personalized email content based on job details and product PRD.

    Args:
        job_details (str): A JSON string containing the extracted job details.
        contact_info (str): A JSON string containing the contact information.
        product_prd (str): The Product Requirements Document (PRD) content.

    Returns:
        str: A JSON string containing the generated email content.

    Example:
        generate_email_content(
            job_details='''{"title": "Software Engineer", "company": "Tech Solutions Inc.", "pay": "$120,000 - $150,000 per year", "duration": "Full-time", "skills": ["Python", "Django", "AWS"]}''',
            contact_info='''{"name": "Jane Doe", "email": "jane.doe@techsolutions.com", "linkedin_profile": "https://www.linkedin.com/in/janedoe"}''',
            product_prd="Our product is an AI-powered coding assistant that helps developers write better code faster."
        )
        # Expected output:
        # {
        #   "job_details": { ... },
        #   "contact_info": { ... },
        #   "product_prd": "...",
        #   "email_content": "Dear Jane, I saw your company is hiring a Software Engineer..."
        # }
    """
    class JobDetailsInput(BaseModel):
        title: str
        company: str
        pay: Optional[str]
        duration: Optional[str]
        skills: List[str]

    class ContactInfoInput(BaseModel):
        name: str
        email: Optional[str]
        linkedin_profile: Optional[str]

    class GraphStateOutput(BaseModel):
        job_details: JobDetailsInput = Field(description="The original job details.")
        contact_info: ContactInfoInput = Field(description="The original contact information.")
        product_prd: str = Field(description="The original product PRD.")
        email_content: str = Field(description="The generated personalized email content.")

    input_str = f"job_details: {job_details}, contact_info: {contact_info}, product_prd: {product_prd}"
    description = generate_email_content.__doc__

    result = llm.with_structured_output(GraphStateOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

def human_approval(email_content: str):
    """
    Presents the generated email content to the user for approval. This node acts as a human-in-the-loop step.

    Args:
        email_content (str): The generated email content to be approved.

    Returns:
        str: A JSON string indicating whether the email content was approved.

    Example:
        human_approval(email_content="Dear Jane, I saw your company is hiring a Software Engineer...")
        # Expected output:
        # {
        #   "email_content": "Dear Jane, I saw your company is hiring a Software Engineer...",
        #   "approved": true
        # }
    """
    class GraphStateOutput(BaseModel):
        email_content: str = Field(description="The email content that was presented for approval.")
        approved: bool = Field(description="True if the email content was approved by the user, False otherwise.")

    input_str = f"email_content: {email_content}"
    description = human_approval.__doc__

    result = llm.with_structured_output(GraphStateOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

def send_email(email_content: str, contact_info: str):
    """
    Sends the personalized email to the identified senior executive after user approval.

    Args:
        email_content (str): The approved email content to be sent.
        contact_info (str): A JSON string containing the recipient's contact information.

    Returns:
        str: A JSON string indicating the email sending status.

    Example:
        send_email(
            email_content="Dear Jane, I saw your company is hiring a Software Engineer...",
            contact_info='''{"name": "Jane Doe", "email": "jane.doe@techsolutions.com", "linkedin_profile": "https://www.linkedin.com/in/janedoe"}'''
        )
        # Expected output:
        # {
        #   "email_content": "Dear Jane, I saw your company is hiring a Software Engineer...",
        #   "contact_info": { ... },
        #   "email_sent_status": true
        # }
    """
    class ContactInfoInput(BaseModel):
        name: str
        email: Optional[str]
        linkedin_profile: Optional[str]

    class GraphStateOutput(BaseModel):
        email_content: str = Field(description="The email content that was sent.")
        contact_info: ContactInfoInput = Field(description="The contact information of the recipient.")
        email_sent_status: bool = Field(description="True if the email was sent successfully, False otherwise.")

    input_str = f"email_content: {email_content}, contact_info: {contact_info}"
    description = send_email.__doc__

    result = llm.with_structured_output(GraphStateOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(MessagesState):
    """
    The GraphState represents the state of the LangGraph workflow.
    It extends MessagesState to include conversation history.
    """
    job_link: Optional[str] = None
    job_details: Optional[dict] = None
    contact_info: Optional[dict] = None
    product_prd: Optional[str] = None
    email_content: Optional[str] = None
    approved: Optional[bool] = None
    email_sent_status: Optional[bool] = None

web_scrape_job_details_tools = [Web_Scraper_Parse_URL_and_Extract_Data]
def web_scrape_job_details(state: GraphState) -> GraphState:
    """
    Node purpose: Extracts job details (title, pay, duration, skills, company) from a given job link.
    Implementation reasoning: Uses a pre-built react agent with the Web Scraper tool to perform web scraping.
    """
    class JobDetailsOutput(BaseModel):
        job_title: str = Field(description="The title of the job.")
        pay: Optional[str] = Field(description="The pay or salary range for the job.", default=None)
        duration: Optional[str] = Field(description="The duration of the job, if specified.", default=None)
        skills: List[str] = Field(description="A list of required skills for the job.")
        company_name: str = Field(description="The name of the company offering the job.")

    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: JobDetailsOutput

    job_link = state.get("job_link")
    if not job_link:
        # Extract job link from the initial human message if not already set
        if state["messages"] and isinstance(state["messages"][0], HumanMessage):
            job_link = state["messages"][0].content
            state["job_link"] = job_link
        else:
            raise ValueError("Job link is missing from the state and initial message.")

    agent = create_react_agent(
      model=llm,
      prompt=f"You are an expert web scraper. Extract the job title, pay, duration, skills, and company name from the following job link: {job_link}. Use the 'Web_Scraper_Parse_URL_and_Extract_Data' tool. If a field is not found, return None for that field.",
      tools=web_scrape_job_details_tools,
      state_schema=CustomStateForReact,
      response_format=JobDetailsOutput
    )

    result: JobDetailsOutput = agent.invoke({"messages" :state["messages"]})["structured_response"]
    
    job_details = result.model_dump()
    
    return {
        "job_details": job_details,
        "messages": [AIMessage(content=f"Job details extracted for {job_details.get('job_title', 'N/A')} at {job_details.get('company_name', 'N/A')}.")]
    }

find_contact_information_tools = [Contact_Finder_Search_and_Retrieve_Contact]
def find_contact_information(state: GraphState) -> GraphState:
    """
    Node purpose: Identifies and retrieves contact information for senior executives within a company based on extracted job details.
    Implementation reasoning: Uses a pre-built react agent with the Contact Finder tool to search for executive contacts.
    """
    class ContactInfoOutput(BaseModel):
        name: str = Field(description="The name of the senior executive.")
        email: Optional[str] = Field(description="The email address of the senior executive.", default=None)
        linkedin_profile: Optional[str] = Field(description="The LinkedIn profile URL of the senior executive.", default=None)
        title: Optional[str] = Field(description="The title of the senior executive.", default=None)

    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: ContactInfoOutput

    job_details = state["job_details"]
    company_name = job_details.get("company_name")
    if not company_name:
        raise ValueError("Company name is missing from job details.")

    agent = create_react_agent(
      model=llm,
      prompt=f"Find contact information (email or LinkedIn profile) for a senior executive at {company_name}. Prioritize roles like CEO, CTO, Head of Product, or similar. Use the 'Contact_Finder_Search_and_Retrieve_Contact' tool.",
      tools=find_contact_information_tools,
      state_schema=CustomStateForReact,
      response_format=ContactInfoOutput
    )

    result: ContactInfoOutput = agent.invoke(state)["structured_response"]
    
    contact_info = result.model_dump()

    return {
        "contact_info": contact_info,
        "messages": [AIMessage(content=f"Contact information found for {contact_info.get('name', 'N/A')} at {company_name}.")]
    }

def generate_email_content(state: GraphState) -> GraphState:
    """
    Node purpose: Generates personalized email content based on job details and product PRD.
    Implementation reasoning: Uses an LLM with structured output to ensure the email content is well-formatted and includes all necessary components.
    """
    class EmailContent(BaseModel):
        subject: str = Field(description="The subject line of the email.")
        body: str = Field(description="The main body of the email, personalized for the recipient and job.")
        call_to_action: str = Field(description="A clear call to action for the recipient.")

    structured_llm = llm.with_structured_output(EmailContent)

    job_details = state["job_details"]
    contact_info = state["contact_info"]
    product_prd = state["product_prd"]
    
    if not job_details or not contact_info or not product_prd:
        # For testing purposes, provide a mock product PRD if not available
        if not product_prd:
            product_prd = "Our product is an AI-powered coding assistant that helps developers write better code faster."
            state["product_prd"] = product_prd
        else:
            raise ValueError("Missing job details, contact info, or product PRD for email generation.")

    prompt = f"""
    Generate a personalized outreach email.
    Recipient Name: {contact_info.get('name', 'Sir/Madam')}
    Recipient Title: {contact_info.get('title', 'Senior Executive')}
    Company: {job_details.get('company_name', 'N/A')}
    Job Title: {job_details.get('job_title', 'N/A')}
    Job Skills: {', '.join(job_details.get('skills', []))}
    Product PRD: {product_prd}

    The email should:
    - Be concise and professional.
    - Reference the job title and company.
    - Briefly explain how our product (based on the PRD) can benefit their company or address a need related to the job.
    - Include a clear call to action.
    """
    
    email_output: EmailContent = structured_llm.invoke([
        HumanMessage(content=prompt),
        AIMessage(content=state["messages"][-1].content)
    ])
    
    full_email_content = f"Subject: {email_output.subject}\n\nDear {contact_info.get('name', 'Sir/Madam')},\n\n{email_output.body}\n\n{email_output.call_to_action}\n\nBest regards,\n[Your Name]"

    return {
        "email_content": full_email_content,
        "messages": [AIMessage(content="Email content generated and ready for human approval.")]
    }

def human_approval(state: GraphState) -> Command[Literal["send_email", "__end__"]]:
    """
    Node purpose: Presents the generated email content to the user for approval. This node acts as a human-in-the-loop step.
    Implementation reasoning: Uses `interrupt` to pause the graph and await human input for approval or rejection.
    """
    email_content = state["email_content"]
    if not email_content:
        raise ValueError("Email content is missing for human approval.")

    approval_decision = interrupt(
        {
            "prompt_for_human": "Please review the generated email content. Do you approve sending this email? (Type 'yes' to approve, 'no' to reject)",
            "email_to_review": email_content
        }
    )
    
    if approval_decision and approval_decision.lower() == "yes":
        return Command(goto="send_email", update={"approved": True, "messages": [AIMessage(content="Email approved by human.")]})
    else:
        return Command(goto="__end__", update={"approved": False, "messages": [AIMessage(content="Email rejected by human. Workflow terminated.")]})

send_email_tools = [Email_Sender_Send_Email]
def send_email(state: GraphState) -> GraphState:
    """
    Node purpose: Sends the personalized email to the identified senior executive after user approval.
    Implementation reasoning: Uses a pre-built react agent with the Email Sender tool to dispatch the email.
    """
    class EmailSendResult(BaseModel):
        success: bool = Field(description="True if the email was sent successfully, False otherwise.")
        message: str = Field(description="A message indicating the outcome of the email sending attempt.")

    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: EmailSendResult

    email_content = state["email_content"]
    contact_info = state["contact_info"]
    
    if not email_content or not contact_info:
        raise ValueError("Missing email content or contact information for sending email.")

    recipient_email = contact_info.get("email")
    if not recipient_email:
        return {
            "email_sent_status": False,
            "messages": [AIMessage(content="Email not sent: Recipient email address not found.")]
        }

    # Extract subject and body from the full email content
    subject_match = re.match(r"Subject: (.*)\n\nDear", email_content, re.DOTALL)
    subject = subject_match.group(1).strip() if subject_match else "No Subject"
    body = email_content[email_content.find("Dear"):].strip()

    agent = create_react_agent(
      model=llm,
      prompt=f"Send an email to {recipient_email} with the subject '{subject}' and the following body: '{body}'. Use the 'Email_Sender_Send_Email' tool.",
      tools=send_email_tools,
      state_schema=CustomStateForReact,
      response_format=EmailSendResult
    )

    result: EmailSendResult = agent.invoke(state)["structured_response"]
    
    return {
        "email_sent_status": result.success,
        "messages": [AIMessage(content=f"Email sending status: {result.message}")]
    }

workflow = StateGraph(GraphState)

workflow.add_node("web_scrape_job_details", web_scrape_job_details)
workflow.add_node("find_contact_information", find_contact_information)
workflow.add_node("generate_email_content", generate_email_content)
workflow.add_node("human_approval", human_approval)
workflow.add_node("send_email", send_email)

workflow.add_edge(START, "web_scrape_job_details")
workflow.add_edge("web_scrape_job_details", "find_contact_information")
workflow.add_edge("find_contact_information", "generate_email_content")
workflow.add_edge("generate_email_content", "human_approval")

workflow.add_edge("send_email", END)

app = workflow.compile(
)
