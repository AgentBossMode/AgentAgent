mock_tools_py =r'''
import requests
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import logging
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# Data models
@dataclass
class ProspectProfile:
    name: str
    company: str
    role: str
    email: str
    linkedin_url: str
    background: str
    pain_points: List[str]
    recent_changes: List[str]

@dataclass
class EmailData:
    to_email: str
    subject: str
    body: str
    template_type: str

@dataclass
class MeetingSlot:
    start_time: datetime
    end_time: datetime
    available: bool
    calendar_link: str

# LinkedIn People Profile Spider Tool
@tool
def LinkedinPeopleProfileSpider(
    search_query: str,
    company_filter: Optional[str] = None,
    role_filter: Optional[str] = None,
    max_profiles: int = 5
) -> Dict[str, Any]:
    """
    Crawls LinkedIn to find people profiles based on search criteria.
    
    Args:
        search_query: Search terms for finding profiles (e.g., "VP Marketing")
        company_filter: Optional company name to filter results
        role_filter: Optional role/title to filter results  
        max_profiles: Maximum number of profiles to return
        
    Returns:
        Dictionary containing found profiles with contact information and background
    """
    try:
        logger.info(f"Searching LinkedIn for: {search_query}")
        
        # Mock LinkedIn search results - in reality, you'd use LinkedIn Sales Navigator API
        # or a web scraping service that complies with LinkedIn's terms
        mock_profiles = [
            {
                "name": "Sarah Johnson",
                "company": "TechCorp Inc",
                "role": "VP of Marketing", 
                "email": "sarah.johnson@techcorp.com",
                "linkedin_url": "https://linkedin.com/in/sarah-johnson-marketing",
                "background": "Former Marketing Director at StartupXYZ, experienced in lead generation and attribution challenges. Recently joined TechCorp to scale their marketing efforts.",
                "pain_points": [
                    "Lead attribution across multiple channels",
                    "Marketing ROI measurement",
                    "Sales and marketing alignment"
                ],
                "recent_changes": [
                    "Recently joined TechCorp as VP Marketing",
                    "Previously struggled with lead attribution at StartupXYZ",
                    "Looking to implement new marketing tech stack"
                ],
                "connection_level": "2nd degree",
                "mutual_connections": 12,
                "last_activity": "Posted about marketing automation 2 days ago"
            },
            {
                "name": "Michael Chen",
                "company": "InnovateCorp",
                "role": "Director of Sales Operations",
                "email": "m.chen@innovatecorp.com", 
                "linkedin_url": "https://linkedin.com/in/michael-chen-sales",
                "background": "Sales operations leader focused on process optimization and CRM implementation. Has experience with Salesforce and HubSpot.",
                "pain_points": [
                    "Sales process inefficiencies",
                    "CRM data quality issues",
                    "Lead scoring accuracy"
                ],
                "recent_changes": [
                    "Recently promoted to Director level",
                    "Leading CRM migration project",
                    "Expanding sales ops team"
                ],
                "connection_level": "3rd degree",
                "mutual_connections": 8,
                "last_activity": "Shared article about sales enablement 1 week ago"
            }
        ]
        
        # Filter results based on criteria
        filtered_profiles = []
        for profile in mock_profiles:
            if company_filter and company_filter.lower() not in profile["company"].lower():
                continue
            if role_filter and role_filter.lower() not in profile["role"].lower():
                continue
            if len(filtered_profiles) >= max_profiles:
                break
            filtered_profiles.append(profile)
        
        # Simulate API delay
        time.sleep(1)
        
        result = {
            "status": "success",
            "profiles_found": len(filtered_profiles), 
            "profiles": filtered_profiles,
            "search_query": search_query,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Found {len(filtered_profiles)} profiles matching criteria")
        return result
        
    except Exception as e:
        logger.error(f"Error in LinkedIn profile search: {str(e)}")
        return {
            "status": "error",
            "error_message": str(e),
            "profiles_found": 0,
            "profiles": []
        }

# Salesforce Email Sender Tool
@tool  
def send_email_via_salesforce(
    to_email: str,
    subject: str,
    body: str,
    from_email: Optional[str] = None,
    template_id: Optional[str] = None,
    track_opens: bool = True,
    track_clicks: bool = True
) -> Dict[str, Any]:
    """
    Sends personalized emails through Salesforce Email API.
    
    Args:
        to_email: Recipient email address
        subject: Email subject line
        body: Email body content (can include HTML)
        from_email: Sender email (optional, uses default if not provided)
        template_id: Salesforce email template ID (optional)
        track_opens: Whether to track email opens
        track_clicks: Whether to track link clicks
        
    Returns:
        Dictionary with send status and tracking information
    """
    try:
        logger.info(f"Sending email via Salesforce to: {to_email}")
        
        # Prepare email data
        email_data = {
            "to_addresses": [to_email],
            "subject": subject,
            "html_body": body,
            "text_body": strip_html_tags(body),
            "from_email": from_email or "sales@yourcompany.com",
            "track_opens": track_opens,
            "track_clicks": track_clicks,
            "timestamp": datetime.now().isoformat()
        }
        
        if template_id:
            email_data["template_id"] = template_id
        
        # Mock Salesforce API call
        # In reality, you'd use the Salesforce REST API or Simple Email Service
        mock_response = {
            "id": f"email_{int(time.time())}",
            "status": "sent",
            "to_email": to_email,
            "subject": subject,
            "sent_at": datetime.now().isoformat(),
            "tracking": {
                "opens": 0,
                "clicks": 0,
                "bounced": False,
                "delivered": True
            },
            "message": "Email sent successfully through Salesforce"
        }
        
        # Simulate API call delay
        time.sleep(0.5)
        
        logger.info(f"Email sent successfully to {to_email}")
        return mock_response
        
    except Exception as e:
        logger.error(f"Error sending email via Salesforce: {str(e)}")
        return {
            "status": "error",
            "error_message": str(e),
            "to_email": to_email,
            "sent_at": None
        }

# Calendar Management Tool
@tool
def list_upcoming_events(
    days_ahead: int = 7,
    calendar_id: Optional[str] = None,
    include_busy_times: bool = True
) -> Dict[str, Any]:
    """
    Lists upcoming calendar events and available time slots.
    
    Args:
        days_ahead: Number of days to look ahead for events
        calendar_id: Specific calendar ID (optional, uses default if not provided)
        include_busy_times: Whether to include busy/blocked time slots
        
    Returns:
        Dictionary with upcoming events and available meeting slots
    """
    try:
        logger.info(f"Fetching calendar events for next {days_ahead} days")
        
        # Generate mock calendar data
        now = datetime.now()
        events = []
        available_slots = []
        
        for day in range(days_ahead):
            current_date = now + timedelta(days=day)
            
            # Mock some existing events
            if day % 2 == 0:  # Every other day has meetings
                events.extend([
                    {
                        "id": f"event_{day}_1",
                        "title": "Team Standup",
                        "start_time": current_date.replace(hour=9, minute=0).isoformat(),
                        "end_time": current_date.replace(hour=9, minute=30).isoformat(),
                        "type": "meeting"
                    },
                    {
                        "id": f"event_{day}_2", 
                        "title": "Client Call",
                        "start_time": current_date.replace(hour=14, minute=0).isoformat(),
                        "end_time": current_date.replace(hour=15, minute=0).isoformat(),
                        "type": "meeting"
                    }
                ])
            
            # Generate available slots (avoiding existing meetings)
            if current_date.weekday() < 5:  # Weekdays only
                morning_slot = {
                    "start_time": current_date.replace(hour=10, minute=0).isoformat(),
                    "end_time": current_date.replace(hour=11, minute=0).isoformat(),
                    "available": True,
                    "slot_type": "morning",
                    "calendar_link": f"https://calendly.com/yourname/30min?date={current_date.strftime('%Y-%m-%d')}&time=10:00"
                }
                
                afternoon_slot = {
                    "start_time": current_date.replace(hour=16, minute=0).isoformat(),
                    "end_time": current_date.replace(hour=17, minute=0).isoformat(),
                    "available": True,
                    "slot_type": "afternoon", 
                    "calendar_link": f"https://calendly.com/yourname/30min?date={current_date.strftime('%Y-%m-%d')}&time=16:00"
                }
                
                available_slots.extend([morning_slot, afternoon_slot])
        
        result = {
            "status": "success",
            "calendar_id": calendar_id or "default",
            "date_range": {
                "start": now.isoformat(),
                "end": (now + timedelta(days=days_ahead)).isoformat()
            },
            "events": events,
            "available_slots": available_slots[:10],  # Limit to 10 slots
            "total_events": len(events),
            "total_available_slots": len(available_slots)
        }
        
        logger.info(f"Found {len(events)} events and {len(available_slots)} available slots")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching calendar events: {str(e)}")
        return {
            "status": "error",
            "error_message": str(e),
            "events": [],
            "available_slots": []
        }

# Utility functions
def strip_html_tags(html_text: str) -> str:
    """Remove HTML tags from text"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', html_text)

def validate_email(email: str) -> bool:
    """Validate email address format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def format_email_template(template: str, variables: Dict[str, str]) -> str:
    """Format email template with variables"""
    formatted = template
    for key, value in variables.items():
        formatted = formatted.replace(f"{{{key}}}", str(value))
    return formatted

# Email templates for different scenarios
EMAIL_TEMPLATES = {
    "cold_outreach": """
    Subject: Quick question about {company}'s marketing attribution challenges

    Hi {name},

    I noticed you recently joined {company} as {role}. Congratulations on the new position!

    I saw from your LinkedIn that you previously dealt with lead attribution challenges at {previous_company}. Many marketing leaders in similar situations struggle with getting clear visibility into which channels are actually driving quality leads.

    We've helped companies like {similar_company} increase their lead attribution accuracy by 40% and improve marketing ROI by 200%. 

    Would you be open to a brief 15-minute conversation to share what we're seeing in the market and how other {role}s are tackling these challenges?

    Best regards,
    {sender_name}
    
    P.S. Here's a quick case study of how we helped a similar company: {case_study_link}
    """
    
    "follow_up": """
    Subject: Following up - {company}'s marketing attribution question

    Hi {name},

    I wanted to follow up on my previous email about marketing attribution challenges.

    I just came across an interesting study showing that 73% of B2B marketers struggle with the same lead attribution issues you mentioned dealing with at {previous_company}.

    Since you're likely evaluating solutions in your new role at {company}, I thought you might find this resource helpful: {resource_link}

    Still interested in that 15-minute conversation? I have a few slots open this week.

    Best,
    {sender_name}
    """
    
    "objection_response": """
    Subject: Re: {company}'s current solution

    Hi {name},

    I completely understand that you're satisfied with {current_solution}. Many of our customers were in the same position before making the switch.

    What I typically find is that companies start looking for alternatives when they need:
    - More advanced attribution modeling beyond last-touch
    - Better integration between sales and marketing data  
    - Deeper insights into customer journey analytics

    Here's a quick comparison showing how we complement solutions like {current_solution}: {comparison_link}

    No pressure at all - but if you're curious to see the differences, I'd be happy to show you a quick 10-minute demo.

    Best regards,
    {sender_name}
    """
}

# Mock data for testing
MOCK_PROSPECT_DATA = {
    "sarah_johnson": {
        "name": "Sarah Johnson",
        "company": "TechCorp Inc", 
        "role": "VP of Marketing",
        "email": "sarah.johnson@techcorp.com",
        "previous_company": "StartupXYZ",
        "pain_points": ["lead attribution", "marketing ROI", "sales alignment"],
        "recent_changes": ["new role", "tech stack evaluation"]
    }
}
'''
main_py = r'''


from langgraph.checkpoint.memory import InMemorySaver
from typing import Dict, Any, List, Optional
import json
import re
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(MessagesState):
    """
    The GraphState represents the state of the LangGraph workflow.
    It extends MessagesState to include conversation history.
    """
    prospects: Optional[List[Dict]] = Field(
        description='A list of identified prospects, each with their contact information and relevant details (e.g., past roles, pain points).'
    )
    outreach_status: Optional[str] = Field(
        default=None,
        description='Status of the outreach (e.g., "sent", "opened", "replied").'
    )
    current_prospect: Optional[Dict] = Field(
        default=None,
        description='The prospect currently being engaged with or for whom the meeting is being scheduled.'
    )
    meeting_details: Optional[Dict] = Field(
        default=None,
        description='Details of the scheduled meeting (e.g., date, time, link).'
    )
    follow_up_status: Optional[str] = Field(
        default=None,
        description='Status of the follow-up (e.g., "sent", "opened").'
    )
    workflow_stage: Optional[str] = Field(
        default="initial",
        description='Current stage of the workflow for better routing decisions.'
    )

# Define tools for each node
prospect_identification_tools = [LinkedinPeopleProfileSpider]
outreach_personalization_tools = [send_email_via_salesforce]
meeting_scheduler_tools = [list_upcoming_events]
follow_up_handler_tools = [send_email_via_salesforce] 

def prospect_identification(state: GraphState) -> GraphState:
    """
    Node purpose: Identifies potential sales prospects by crawling LinkedIn and other web pages for contact information.
    Implementation reasoning: Uses a react agent with a web crawling tool to perform the identification,
                            as this requires dynamic tool use and reasoning.
    """
    agent = create_react_agent(
        model=llm,
        tools=prospect_identification_tools,
        prompt="You are an expert sales researcher. Your task is to identify potential sales prospects based on the user's input. Use the LinkedIn profile spider tool to gather contact information and relevant details (e.g., past roles, pain points, recent job changes) from LinkedIn profiles. Focus on extracting actionable prospect data including name, email, company, role, and any relevant background information that could be used for personalized outreach."
    )
    
    # Get the initial user input
    user_input = state["messages"][-1].content if state["messages"] else ""
    
    try:
        response = agent.invoke({"messages": state["messages"]})
        
        # Parse prospect data from the agent response
        identified_prospects = []
        content = response["messages"][-1].content
        
        # Try to extract structured prospect data
        # Look for JSON-like structures or parse text for prospect information
        json_match = re.search(r'\[.*\]|\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                parsed_data = json.loads(json_str)
                if isinstance(parsed_data, list):
                    identified_prospects = parsed_data
                elif isinstance(parsed_data, dict):
                    identified_prospects = [parsed_data]
            except json.JSONDecodeError:
                pass
        
        # If no structured data found, create a prospect from the response
        if not identified_prospects:
            # Extract key information from the response text
            prospect_data = {
                "name": extract_name_from_text(content),
                "company": extract_company_from_text(content),
                "role": extract_role_from_text(content),
                "email": extract_email_from_text(content),
                "background": content[:500], 
                "source": "linkedin_crawl"
            }
            identified_prospects = [prospect_data]
        
        return {
            "prospects": identified_prospects,
            "current_prospect": identified_prospects[0] if identified_prospects else None,
            "messages": state["messages"] + [response["messages"][-1]],
            "workflow_stage": "prospects_identified"
        }
    
    except Exception as e:
        error_prospect = {
            "name": "Unknown",
            "error": f"Failed to identify prospects: {str(e)}",
            "source": "error"
        }
        return {
            "prospects": [error_prospect],
            "current_prospect": error_prospect,
            "messages": state["messages"] + [AIMessage(content=f"Error in prospect identification: {str(e)}")],
            "workflow_stage": "error"
        }

def outreach_personalization(state: GraphState) -> GraphState:
    """
    Node purpose: Personalizes cold outreach emails or messages based on the identified prospects' information.
    Implementation reasoning: Uses a react agent with an email sending tool to craft and send personalized outreach.
                            The agent can reason about the best personalization strategy.
    """
    agent = create_react_agent(
        model=llm,
        tools=outreach_personalization_tools,
        prompt="""You are an expert sales outreach specialist. Your task is to personalize and send cold outreach emails to prospects."""

Key requirements:
1. Reference the prospect's past roles, company changes, and pain points
2. Keep the message concise and value-focused
3. Include a clear call-to-action
4. Use the send_email_via_salesforce tool to actually send the email
5. Personalize based on recent job changes, industry trends, or company news

The email should feel personal and relevant, not generic. Handle any concerns or questions professionally and focus on scheduling a meeting or demo."""
)
    
    current_prospect = state["current_prospect"]
    if not current_prospect and state["prospects"]:
        current_prospect = state["prospects"][0]
    
    if not current_prospect:
        return {
            "messages":[AIMessage(content="No current prospect available for outreach personalization.")],
            "outreach_status": "failed",
            "workflow_stage": "outreach_failed"
        }
    
    try:
        # Create a detailed context for the agent
        prospect_context = f"""
        Prospect Details:
        - Name: {current_prospect.get('name', 'Unknown')}
        - Company: {current_prospect.get('company', 'Unknown')}
        - Role: {current_prospect.get('role', 'Unknown')}
        - Email: {current_prospect.get('email', 'Unknown')}
        - Background: {current_prospect.get('background', 'No background available')}
        
        Craft and send a personalized outreach email that references their background and offers relevant value.
        If there are any concerns or questions in previous messages, address them professionally and focus on the value proposition.
        """
        
        response = agent.invoke({
            "messages": [HumanMessage(content=prospect_context)]
        })
        
        return {
            "outreach_status": "sent",
            "current_prospect": current_prospect,
            "workflow_stage": "outreach_sent"
        }
    
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error in outreach personalization: {str(e)}")],
            "outreach_status": "failed",
            "current_prospect": current_prospect,
            "workflow_stage": "outreach_failed"
        }

def meeting_scheduler(state: GraphState) -> GraphState:
    """
    Node purpose: Books meetings with interested prospects.
    Implementation reasoning: Uses a react agent with a calendar management tool to handle meeting scheduling.
                            The agent can interact with the tool to find available slots and send invites.
    """
    agent = create_react_agent(
        model=llm,
        tools=meeting_scheduler_tools,
        prompt="""You are an expert meeting scheduler. Your task is to book meetings with interested prospects.

Steps to follow:
1. Use the list_upcoming_events tool to check calendar availability
2. Propose 2-3 time slots that work
3. Send a calendar invite with meeting details
4. Include relevant meeting agenda items based on the prospect's background

Focus on finding mutually convenient times and making the meeting value-driven."""
    )
    
    current_prospect = state["current_prospect"]
    if not current_prospect:
        return {
            "messages": [AIMessage(content="No current prospect available for meeting scheduling.")],
            "meeting_details": {"status": "failed"},
            "workflow_stage": "meeting_failed"
        }
    
    try:
        meeting_context = f"""
        Schedule a meeting with:
        - Name: {current_prospect.get('name', 'Unknown')}
        - Company: {current_prospect.get('company', 'Unknown')}
        - Role: {current_prospect.get('role', 'Unknown')}
        - Email: {current_prospect.get('email', 'Unknown')}
        
        The prospect has shown interest. Find available time slots and send a meeting invite.
        """
        
        response = agent.invoke({
            "messages": [HumanMessage(content=meeting_context)]
        })
        
        meeting_details = {
            "status": "scheduled",
            "prospect": current_prospect,
        }
        
        return {
            "meeting_details": meeting_details,
            "workflow_stage": "meeting_scheduled"
        }
    
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error in meeting scheduling: {str(e)}")],
            "meeting_details": {"status": "failed", "error": str(e)},
            "workflow_stage": "meeting_failed"
        }

def follow_up_handler(state: GraphState) -> GraphState:
    """
    Node purpose: Handles automated follow-ups for unopened outreach emails after a specified period.
    Implementation reasoning: Uses a react agent with an email sending tool to send automated follow-ups.
                            The agent can determine the best follow-up message based on the context.
    """
    agent = create_react_agent(
        model=llm,
        tools=follow_up_handler_tools,
        prompt="""You are an expert sales follow-up specialist. Your task is to send thoughtful follow-up emails.

Follow-up best practices:
1. Reference the previous outreach without being pushy
2. Add new value (industry insight, case study, resource)
3. Keep it brief and focused
4. Include a different call-to-action than the original email
5. Use the send_email_via_salesforce tool to send the follow-up

Make the follow-up feel helpful, not salesy. Address any concerns professionally while highlighting benefits."""
    )
    
    current_prospect = state["current_prospect"]
    if not current_prospect:
        return {
            "messages": [AIMessage(content="No current prospect available for follow-up.")],
            "follow_up_status": "failed",
            "workflow_stage": "followup_failed"
        }
    
    try:
        followup_context = f"""
        Send a follow-up email to:
        - Name: {current_prospect.get('name', 'Unknown')}
        - Company: {current_prospect.get('company', 'Unknown')}
        - Role: {current_prospect.get('role', 'Unknown')}
        - Email: {current_prospect.get('email', 'Unknown')}
        
        Previous outreach status: {state["outreach_status"]}
        
        Craft a helpful follow-up that adds new value and doesn't repeat the previous message.
        If there were any concerns or questions in previous messages, address them professionally.
        """
        
        response = agent.invoke({
            "messages": state["messages"] + [HumanMessage(content=followup_context)]
        })
        
        return {
            "follow_up_status": "sent",
            "workflow_stage": "followup_sent"
        }
    
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error in follow-up: {str(e)}")],
            "follow_up_status": "failed",
            "workflow_stage": "followup_failed"
        }

# Routing functions
def route_outreach_response(state: GraphState) -> str:
    """
    Routing function for outreach_personalization node.
    Routes based on prospect's response or outreach status.
    """
    if not state["messages"]:
        return "follow_up_handler"
    
    last_message_content = state["messages"][-1].content.lower()
    
    # Check for positive interest indicators
    positive_indicators = ["interested", "schedule", "meeting", "demo", "call", "yes", "sounds good", "tell me more"]
    if any(indicator in last_message_content for indicator in positive_indicators):
        return "meeting_scheduler"
    
    # For any concerns, objections, or neutral responses, continue with follow-up
    # The follow-up handler will address concerns professionally
    return "follow_up_handler"

def route_initial_input(state: GraphState) -> str:
    """
    Routing function for the initial input.
    Always routes to prospect_identification to start the sales process.
    """
    return "prospect_identification"

# Helper functions for data extraction
def extract_name_from_text(text: str) -> str:
    """Extract name from text using simple heuristics."""
    # Look for "Name: John Doe" pattern
    name_match = re.search(r'name[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)', text, re.IGNORECASE)
    if name_match:
        return name_match.group(1)
    return "Unknown"

def extract_company_from_text(text: str) -> str:
    """Extract company from text using simple heuristics."""
    company_match = re.search(r'company[:\s]+([A-Z][A-Za-z\s&]+)', text, re.IGNORECASE)
    if company_match:
        return company_match.group(1).strip()
    return "Unknown"

def extract_role_from_text(text: str) -> str:
    """Extract role/title from text using simple heuristics."""
    role_patterns = [
        r'(?:role|title|position)[:\s]+([A-Z][A-Za-z\s]+)',
        r'(VP|Director|Manager|CEO|CTO|CMO|Head of|Senior)',
    ]
    for pattern in role_patterns:
        role_match = re.search(pattern, text, re.IGNORECASE)
        if role_match:
            return role_match.group(1).strip()
    return "Unknown"

def extract_email_from_text(text: str) -> str:
    """Extract email from text using regex."""
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        return email_match.group(0)
    return "Unknown"

# Build the workflow graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("prospect_identification", prospect_identification)
workflow.add_node("outreach_personalization", outreach_personalization)
workflow.add_node("meeting_scheduler", meeting_scheduler)
workflow.add_node("follow_up_handler", follow_up_handler)

# Add edges
workflow.add_edge(START,"prospect_identification")

workflow.add_edge("prospect_identification", "outreach_personalization")

workflow.add_conditional_edges(
    "outreach_personalization",
    route_outreach_response,
    {
        "meeting_scheduler": "meeting_scheduler",
        "follow_up_handler": "follow_up_handler"
    }
)

workflow.add_edge("meeting_scheduler", END)
workflow.add_edge("follow_up_handler", END)

# Compile the workflow
checkpointer = InMemorySaver()
# app = workflow.compile(checkpointer=checkpointer)
app =  workflow.compile(checkpointer)
'''


fitness_tools = r'''
from langchain_core.tools import tool
from datetime import datetime
import requests

@tool
def FoodCalorieCalculator(food_item: str, quantity: float, unit: str) -> float:
    """
    A tool to calculate the calorie content of a given food item.
    Input: food item (str), quantity (float), unit (str).
    Output: estimated calories (float).
    """
    if food_item.lower() == "apple":
        calories_per_unit = 52  
        if unit.lower() == "g":
            return (quantity / 100) * calories_per_unit
        elif unit.lower() == "piece":
            return quantity * 95
    elif food_item.lower() == "chicken breast":
        calories_per_unit = 165 
        if unit.lower() == "g":
            return (quantity / 100) * calories_per_unit
        elif unit.lower() == "piece":
            return quantity * 250
    return 0.0


@tool
def StoreFoodIntake(database_id: str, properties: list, child_blocks: list = None, cover: str = None, icon: str = None) -> dict:
    """
    Creates a new page (row) in a specified notion database.
    """
    print(f"Mock: Storing food intake in database {database_id} with properties {properties}")
    return {"data": {"response_data": {"id": "mock_page_id", "properties": properties}}, "successful": True}

@tool
def ExerciseCalorieEstimator(exercise_type: str, duration: float, unit: str, user_weight: float = None) -> float:
    """
    A tool to estimate calories burned for a given exercise.
    Input: exercise type (str), duration (float), unit (str), user weight (optional, float).
    Output: estimated calories burned (float).
    """
    calories_per_minute = {
        "running": 10,
        "cycling": 8,
        "swimming": 7,
        "walking": 4
    }
    
    duration_minutes = duration
    if unit.lower() == "hours":
        duration_minutes = duration * 60
    
    estimated_calories = calories_per_minute.get(exercise_type.lower(), 5) * duration_minutes
    
    if user_weight:
        estimated_calories *= (user_weight / 150)
        
    return estimated_calories

@tool
def StoreExerciseData(database_id: str, properties: list, child_blocks: list = None, cover: str = None, icon: str = None) -> dict:
    """
    Creates a new page (row) in a specified notion database.
    """
    print(f"Mock: Storing exercise data in database {database_id} with properties {properties}")
    return {"data": {"response_data": {"id": "mock_page_id", "properties": properties}}, "successful": True}

@tool
def RetrieveDietHistory(database_id: str, page_size: int = 2, sorts: list = None, start_cursor: str = None) -> dict:
    """
    Queries a notion database for pages (rows), where rows are pages and columns are properties; ensure sort property names correspond to existing database properties.
    """
    print(f"Mock: Retrieving diet history from database {database_id}")
    mock_data = {
        "results": [
            {"properties": {"Food Item": {"title": [{"plain_text": "Apple"}]}, "Calories": {"number": 95}}},
            {"properties": {"Food Item": {"title": [{"plain_text": "Chicken Breast"}]}, "Calories": {"number": 250}}}
        ],
        "next_cursor": None,
        "has_more": False
    }
    return {"data": {"response_data": mock_data}, "successful": True}


@tool
def RetrieveExerciseHistory(database_id: str, page_size: int = 2, sorts: list = None, start_cursor: str = None) -> dict:
    """
    Queries a notion database for pages (rows), where rows are pages and columns are properties; ensure sort property names correspond to existing database properties.
    """
    print(f"Mock: Retrieving exercise history from database {database_id}")
    mock_data = {
        "results": [
            {"properties": {"Exercise Type": {"title": [{"plain_text": "Running"}]}, "Calories Burned": {"number": 300}}},
            {"properties": {"Exercise Type": {"title": [{"plain_text": "Cycling"}]}, "Calories Burned": {"number": 400}}}
        ],
        "next_cursor": None,
        "has_more": False
    }
    return {"data": {"response_data": mock_data}, "successful": True}

@tool
def RetrieveTotalCaloriesConsumed(start_date: datetime, end_date: datetime, user) -> float:
    """
    A tool to retrieve total calories consumed for a specified period from the database.
    Input: start_date (datetime), end_date (datetime), user (object representing the user).
    Output: total calories consumed (float).
    """
    class MockFoodItem:
        def __init__(self, calorie):
            self.calorie = calorie

    class MockUserFoodItem:
        def __init__(self, food_items, date_consumed):
            self.fooditem = food_items
            self.date_consumed = date_consumed

    mock_consumed_items = [
        MockUserFoodItem([MockFoodItem(150), MockFoodItem(200)], datetime(2023, 1, 10)),
        MockUserFoodItem([MockFoodItem(300)], datetime(2023, 1, 11)),
        MockUserFoodItem([MockFoodItem(100), MockFoodItem(50)], datetime(2023, 1, 12)),
    ]

    total_calories = 0
    for item in mock_consumed_items:
        if start_date <= item.date_consumed <= end_date:
            for food in item.fooditem:
                total_calories += food.calorie
    return float(total_calories)

@tool
def RetrieveTotalCaloriesBurned(exercise_type: str, duration_minutes: int = 60, weight: int = 160, api_key: str = 'YOUR_API_KEY') -> float:
    """
    A tool to retrieve total calories burned for a specified period from the database.
    Input: exercise_type (str), duration_minutes (int, default 60), weight (int, default 160), api_key (str).
    Output: total calories burned (float).
    """
    url = f'https://api.api-ninjas.com/v1/caloriesburned?activity={exercise_type}&duration={duration_minutes}&weight={weight}'
    headers = {'X-Api-Key': api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        total_calories = data[0]['total_calories'] if data else 0
        return float(total_calories)
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
'''

fitness_main = r'''
from typing import Optional
import uuid
from typing import Dict, Any
from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from typing import Dict, Any, List, Optional, Literal, TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
import re
import json
import os
from datetime import datetime
from tools_code import FoodCalorieCalculator, StoreFoodIntake, ExerciseCalorieEstimator, StoreExerciseData, RetrieveDietHistory, RetrieveExerciseHistory, RetrieveTotalCaloriesConsumed, RetrieveTotalCaloriesBurned


# Create llm definition
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(MessagesState):
    """
    Represents the state of the LangGraph workflow.
    The 'messages' field is inherited from MessagesState.
    """
    route: Optional[str] = None
    food_calories_tracked: Optional[bool] = None
    exercise_calories_tracked: Optional[bool] = None
    diet_history_retrieved: Optional[str] = None
    exercise_history_retrieved: Optional[str] = None
    net_calories_retrieved: Optional[str] = None

class RouteQueryOutput(BaseModel):
    """Structured output for routing the user query."""
    route: Literal["track_food_calories", "track_exercise_calories", "get_diet_history", "get_exercise_history", "get_net_calories", "unsupported"] = Field(description="The identified intent of the user query.")
    reasoning: str = Field(description="Brief explanation for the routing decision.")

def route_query(state: GraphState) -> GraphState:
    """
    Node purpose: Routes the user query to the appropriate node based on its intent.
    Implementation reasoning: Uses structured output to ensure a clear and type-safe routing decision.
    """
    structured_llm = llm.with_structured_output(RouteQueryOutput)
    user_message = state["messages"][-1].content

    prompt = f"""
    Analyze the user's query and determine its primary intent.
    The query is: "{user_message}"

    Possible intents are:
    - 'track_food_calories': User wants to log food intake and calculate calories.
    - 'track_exercise_calories': User wants to log exercise and calculate calories burned.
    - 'get_diet_history': User wants to retrieve past food consumption data.
    - 'get_exercise_history': User wants to retrieve past exercise data.
    - 'get_net_calories': User wants to know their net calorie balance (consumed vs. burned).
    - 'unsupported': The query does not fit any of the above categories.

    Provide the most relevant intent and a brief reasoning.
    """

    result = structured_llm.invoke([HumanMessage(content=prompt)])

    return {
        "route": result.route,
        "messages": [AIMessage(content=f"Query routed to: {result.route}. Reasoning: {result.reasoning}")]
    }

track_food_calories_tools = [FoodCalorieCalculator, StoreFoodIntake]
def track_food_calories(state: GraphState) -> GraphState:
    """
    Node purpose: Calculates and stores calories consumed from user-reported food intake.
    Implementation reasoning: Uses a react agent to intelligently use the FoodCalorieCalculator and StoreFoodIntake tools.
    """
    agent = create_react_agent(
        model=llm,
        tools=track_food_calories_tools,
        prompt="You are an AI assistant that helps track food calorie intake. Use the provided tools to calculate calories for food items and then store the intake data. Extract food item, quantity, and unit from the user's message. If a timestamp is not provided, use the current time. Always respond to the user about the outcome."
    )
    response = agent.invoke({"messages": state["messages"]})
    return {
        "food_calories_tracked": True,
        "messages": response["messages"]
    }

track_exercise_calories_tools = [ExerciseCalorieEstimator, StoreExerciseData]
def track_exercise_calories(state: GraphState) -> GraphState:
    """
    Node purpose: Calculates and stores calories burned from user-reported exercise.
    Implementation reasoning: Uses a react agent to intelligently use the ExerciseCalorieEstimator and StoreExerciseData tools.
    """
    agent = create_react_agent(
        model=llm,
        tools=track_exercise_calories_tools,
        prompt="You are an AI assistant that helps track exercise calories. Use the provided tools to estimate calories burned for exercises and then store the exercise data. Extract exercise type, duration, and unit from the user's message. If a user weight is not provided, assume an average. If a timestamp is not provided, use the current time. Always respond to the user about the outcome."
    )
    response = agent.invoke({"messages": state["messages"]})
    return {
        "exercise_calories_tracked": True,
        "messages": response["messages"]
    }

get_diet_history_tools = [RetrieveDietHistory]
def get_diet_history(state: GraphState) -> GraphState:
    """
    Node purpose: Retrieves and answers questions about past food consumption from the database.
    Implementation reasoning: Uses a react agent to intelligently use the RetrieveDietHistory tool.
    """
    agent = create_react_agent(
        model=llm,
        tools=get_diet_history_tools,
        prompt="You are an AI assistant that retrieves diet history. Use the RetrieveDietHistory tool to fetch past food consumption data. Extract the date or time frame from the user's message. If no specific date is given, retrieve for today. Summarize the retrieved information for the user."
    )
    response = agent.invoke({"messages": state["messages"]})
    return {
        "diet_history_retrieved": response["messages"][-1].content,
        "messages": response["messages"]
    }

get_exercise_history_tools = [RetrieveExerciseHistory]
def get_exercise_history(state: GraphState) -> GraphState:
    """
    Node purpose: Retrieves and answers questions about past workouts from the database.
    Implementation reasoning: Uses a react agent to intelligently use the RetrieveExerciseHistory tool.
    """
    agent = create_react_agent(
        model=llm,
        tools=get_exercise_history_tools,
        prompt="You are an AI assistant that retrieves exercise history. Use the RetrieveExerciseHistory tool to fetch past workout data. Extract the date or time frame from the user's message. If no specific date is given, retrieve for today. Summarize the retrieved information for the user."
    )
    response = agent.invoke({"messages": state["messages"]})
    return {
        "exercise_history_retrieved": response["messages"][-1].content,
        "messages": response["messages"]
    }

get_net_calories_tools = [RetrieveTotalCaloriesConsumed, RetrieveTotalCaloriesBurned]
def get_net_calories(state: GraphState) -> GraphState:
    """
    Node purpose: Provides insights into the overall calorie balance (consumed vs. burned) from the database.
    Implementation reasoning: Uses a react agent to intelligently use RetrieveTotalCaloriesConsumed and RetrieveTotalCaloriesBurned tools.
    """
    agent = create_react_agent(
        model=llm,
        tools=get_net_calories_tools,
        prompt="You are an AI assistant that calculates net calorie balance. Use the RetrieveTotalCaloriesConsumed and RetrieveTotalCaloriesBurned tools to get the necessary data. Extract the date or time frame from the user's message. If no specific date is given, calculate for today. Provide a clear summary of the net calorie effect to the user."
    )
    response = agent.invoke({"messages": state["messages"]})
    return {
        "net_calories_retrieved": response["messages"][-1].content,
        "messages": response["messages"]
    }

def route_decision(state: GraphState) -> str:
    """
    Routing function for the 'route_query' node.
    Determines the next node based on the 'route' field in the state.
    """
    print(f"Routing based on: {state['route']}")
    return state["route"]

# Build the graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("route_query", route_query)
workflow.add_node("track_food_calories", track_food_calories)
workflow.add_node("track_exercise_calories", track_exercise_calories)
workflow.add_node("get_diet_history", get_diet_history)
workflow.add_node("get_exercise_history", get_exercise_history)
workflow.add_node("get_net_calories", get_net_calories)
# Set entry point
workflow.add_edge(START, "route_query")

# Add conditional edges from route_query
workflow.add_conditional_edges(
    "route_query",
    route_decision,
    {
        "track_food_calories": "track_food_calories",
        "track_exercise_calories": "track_exercise_calories",
        "get_diet_history": "get_diet_history",
        "get_exercise_history": "get_exercise_history",
        "get_net_calories": "get_net_calories",
        "unsupported": END # If the intent is unsupported, end the graph
    }
)

# Add non-conditional edges to END
workflow.add_edge("track_food_calories", END)
workflow.add_edge("track_exercise_calories", END)
workflow.add_edge("get_diet_history", END)
workflow.add_edge("get_exercise_history", END)
workflow.add_edge("get_net_calories", END)
# Compile the graph
app = workflow.compile()
'''

pytest = r'''
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
        ("I ate a banana", "I logged your intake of 1 medium banana, which is estimated to have 0 calories. If you have more food items to track, just let me know!"),
("I ran for 30 minutes", "Exercise calories stored successfully."),
("What did I eat on Monday?", "It seems I retrieved some of your recent food consumption data, but it doesnt specifically indicate what you ate on Monday. Heres a summary of what I found: 1. Apple - 95 calories 2. Chicken Breast - 250 calories If you need more specific details or a different time frame, please let me know!"),
("How many calories did I burn yesterday?", "Yesterday, you burned a total of y calories from your workouts:- **Running**: x calories - **Cycling**: z calories. If you need more details or information about other days, feel free to ask!"),
("What's my net calorie intake for the week?", " It seems that there was an issue retrieving your total calories consumed for the week, as the data returned was zero. Additionally, I was unable to retrieve the calories burned."),
    ],
)
def test_full_workflow_final_response(input_query: str, reference_output: str):
    # Invoke the full graph
    thread_config = {"configurable": {"thread_id": uuid4() }}
    result = app.invoke({"messages": [HumanMessage(content=input_query)]}, config=thread_config)

    # Get the last message, which is the final response
    actual_output = result["messages"][-1].content
    assert True == True
    



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
     ("I ate a banana", ['route_query', 'track_food_calories']),
("I ran for 30 minutes", ['route_query', 'track_exercise_calories']),
("What did I eat on Monday?", ['route_query', 'get_diet_history']),
("How many calories did I burn yesterday?", ['route_query', 'get_exercise_history']),
("What's my net calorie intake for the week?", ['route_query', 'get_net_calories'])
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
        if chunk['type'] == 'task':
            if chunk['payload']['name'] != "agent" and chunk['payload']['name'] != "tools":
                trajectory.append(chunk['payload']['name'])
    
    grading_assignment = f"""QUESTION: {input_query}
    GROUND TRUTH TRAJECTORY: {" ".join(expected_tool_call_names)}
    STUDENT TRAJECTORY: {" ".join(trajectory)}"""
    grade: GradeTrajectory = grade_trajectory_llm.invoke([{"role": "system", "content": grader_trajectory_instructions}, {"role": "user", "content": grading_assignment}])
    return grade.is_correct
'''