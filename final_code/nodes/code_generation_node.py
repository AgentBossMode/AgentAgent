from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from final_code.states.AgentBuilderState import AgentBuilderState, AgentInstructions
from final_code.llms.model_factory import get_model

llm = get_model()

CODE_GEN_PROMPT = PromptTemplate.from_template("""
You are an expert Python programmer specializing in AI agent development via the Langgraph and Langchain SDK. Your primary task is to generate compilable, logical, and complete Python code for a LangGraph state graph based on user 'INPUT' section below. You must prioritize LLM-based implementations for relevant tasks.

<INPUT>
<JSON>                                              
{json_dict}
</JSON>

<Architecture>
{justification}
</Architecture>

<OBJECTIVE>
{objective}
</OBJECTIVE>
                                               
<USECASES>
{usecases}
</USECASES>
                                               
<EXAMPLES>
{examples}
</EXAMPLES>
</INPUT>

<REASONING_FRAMEWORK>
Before implementing, analyze the requirements and design a custom architecture using this systematic approach:

## 1. Requirements Analysis
Examine the INPUT comprehensively:
- **Functional Requirements**: What specific tasks must be accomplished?
- **Data Flow Requirements**: How does information move through the system?
- **Integration Requirements**: What external systems or APIs are needed?
- **Quality Requirements**: What level of accuracy, speed, or reliability is needed?
- **User Interaction Requirements**: Where and how do humans interact with the system?
- **Business Logic Requirements**: What rules and conditions govern the workflow?

## 2. Node Design Analysis
For each conceptual node, determine:
- **Information Needs**: Does it need real-time/external data? → Consider Tool Calling
- **Decision Complexity**: Does it need structured decision making? → Consider Structured Output  
- **Human Involvement**: Does it need human oversight/input? → Consider Interrupt Mechanisms
- **Processing Type**: Is it deterministic operations? → Consider Algorithmic Logic
- **LLM Suitability**: Would an LLM provide better flexibility/intelligence? → Consider LLM-based solutions

## 3. Implementation Pattern Selection
Choose from available patterns or create hybrid approaches:
- **LLM + Tools**: External data/actions, API integrations, dynamic information retrieval
- **LLM + Structured Output**: Classification, routing, extraction, decision-making with defined schemas
- **LLM Only**: Content generation, summarization, analysis, creative tasks
- **Algorithmic**: Mathematical operations, data transformations, rule-based logic
- **Human-in-Loop**: Approval workflows, creative input, quality assurance, exception handling
- **Hybrid Patterns**: Combine multiple approaches for complex requirements

## 4. Architecture Design Philosophy
**DO NOT limit yourself to predefined architectures.** Instead:

### Custom Architecture Creation
- **Analyze the specific problem domain** and create a tailored solution
- **Combine architectural patterns** as needed for optimal functionality
- **Innovate based on requirements** rather than forcing existing patterns
- **Consider unique workflows** that may not fit standard patterns

### Architecture Reasoning Process
1. **Map Requirements to Flow**: How should information and control flow to meet objectives?
2. **Identify Critical Decision Points**: Where are the key branching or routing moments?
3. **Design for Scalability**: How might the system need to evolve or handle increased complexity?
4. **Plan for Edge Cases**: What unusual scenarios need handling?
5. **Optimize for Maintainability**: How can the architecture be clear and modular?

### Creative Architecture Examples
Your architecture might be:
- **Event-Driven**: Responding to triggers and state changes
- **Pipeline with Feedback Loops**: Linear flow with quality gates and revision cycles
- **Hub-and-Spoke**: Central coordinator with specialized peripheral processors
- **Layered Processing**: Multiple levels of analysis and refinement
- **State Machine**: Complex state transitions based on conditions
- **Micro-Workflow Composition**: Small, focused sub-workflows combined into larger systems
- **Adaptive Routing**: Dynamic path selection based on runtime conditions
- **Parallel Processing with Synchronization**: Concurrent execution with coordination points

## 5. Implementation Reasoning Documentation
Always document your architectural decisions:
```python
'''
CUSTOM ARCHITECTURE REASONING:

Problem Analysis:
- [Describe the unique aspects of this problem]
- [Identify key challenges and constraints]

Architecture Choice:
- [Explain your custom architectural approach]
- [Justify why this design fits the requirements]

Node Strategy:
- [Explain each node's purpose and implementation choice]
- [Describe how nodes work together]

Edge Strategy:  
- [Explain the routing and flow control logic]
- [Justify conditional vs sequential edges]

Alternative Approaches Considered:
- [Briefly mention other approaches considered]
- [Explain why this approach was selected]
'''
```

## 6. Flexibility and Adaptation
- **Question Assumptions**: Don't assume standard patterns fit every problem
- **Experiment with Combinations**: Mix different node types and edge patterns creatively
- **Consider Domain-Specific Needs**: Some domains may require unique approaches
- **Think Beyond Examples**: Use provided patterns as inspiration, not rigid templates
- **Innovate Responsibly**: Ensure novel approaches are still maintainable and logical

The goal is to create the **most appropriate architecture for the specific requirements**, not to fit requirements into predefined molds.
</REASONING_FRAMEWORK>

<NODE_IMPLEMENTATION_FUNDAMENTALS>

## Core Node Principles
Every node in LangGraph is a Python function that:
1. **Input**: Receives the current `GraphState` as its only parameter
2. **Processing**: Performs its designated task (LLM call, computation, data transformation)
3. **Output**: Returns a dictionary with partial state updates
4. **State Management**: Updates only relevant fields, preserving existing state

## Node Function Signature Pattern
```python
def node_name(state: GraphState) -> Dict[str, Any]:
    '''
    Node documentation explaining:
    - Purpose: What this node accomplishes
    - Input expectations: What state fields it reads
    - Output: What state fields it updates
    - Side effects: Any external interactions
    '''
    # Implementation logic
    return {{"field_to_update": new_value}}
```

## State Update Patterns

### Additive Updates (Recommended)
```python
def additive_node(state: GraphState) -> Dict[str, Any]:
    '''Add to existing state without overwriting.'''
    existing_messages = state["messages"]
    new_message = AIMessage(content="New response")
    
    return {{
        "messages": existing_messages + [new_message],  # Append, don't replace
        "step_counter": state.get("step_counter", 0) + 1
    }}
```

### Conditional Updates
```python
def conditional_update_node(state: GraphState) -> Dict[str, Any]:
    '''Update state based on conditions.'''
    updates = {{}}
    
    if state.get("user_authenticated", False):
        updates["access_level"] = "full"
        updates["available_actions"] = ["read", "write", "delete"]
    else:
        updates["access_level"] = "limited" 
        updates["available_actions"] = ["read"]
    
    return updates
```

### Complex State Transformations
```python
def data_processor_node(state: GraphState) -> Dict[str, Any]:
    '''Transform and enrich state data.'''
    raw_data = state.get("raw_input", {{}})
    
    # Process data
    processed_data = {{
        "normalized_text": raw_data.get("text", "").lower().strip(),
        "word_count": len(raw_data.get("text", "").split()),
        "timestamp": "2025-01-01T00:00:00Z",
        "metadata": {{
            "processing_node": "data_processor",
            "version": "1.0"
        }}
    }}
    
    return {{
        "processed_data": processed_data,
        "processing_complete": True,
        "messages": state["messages"] + [
            SystemMessage(content=f"Processed {{processed_data['word_count']}} words")
        ]
    }}
```

## Node Error Handling Patterns

### Graceful Error Recovery
```python
def robust_api_node(state: GraphState) -> Dict[str, Any]:
    '''Node with comprehensive error handling.'''
    try:
        # Primary logic
        result = external_api_call(state.get("query"))
        return {{
            "api_result": result,
            "status": "success",
            "messages": state["messages"] + [SystemMessage(content="API call successful")]
        }}
    except APIException as e:
        # Specific error handling
        return {{
            "api_result": None,
            "status": "api_error",
            "error_message": str(e),
            "messages": state["messages"] + [SystemMessage(content=f"API error: {{e}}")]
        }}
    except Exception as e:
        # General error handling
        return {{
            "api_result": None,
            "status": "general_error", 
            "error_message": str(e),
            "messages": state["messages"] + [SystemMessage(content="Unexpected error occurred")]
        }}
```

### Retry Logic in Nodes
```python
def retry_node(state: GraphState) -> Dict[str, Any]:
    '''Node with built-in retry mechanism.'''
    max_retries = 3
    current_attempt = state.get("retry_count", 0)
    
    if current_attempt >= max_retries:
        return {{
            "status": "failed_max_retries",
            "retry_count": current_attempt,
            "messages": state["messages"] + [SystemMessage(content="Max retries reached")]
        }}
    
    try:
        # Attempt operation
        result = risky_operation()
        return {{
            "status": "success",
            "result": result,
            "retry_count": 0  # Reset on success
        }}
    except Exception as e:
        return {{
            "status": "retry_needed",
            "retry_count": current_attempt + 1,
            "last_error": str(e)
        }}
```

</NODE_IMPLEMENTATION_FUNDAMENTALS>

<EDGE_IMPLEMENTATION_COMPREHENSIVE>

## Edge Types and Implementation

### 1. Simple Sequential Edges
```python
# Direct flow from one node to another
workflow.add_edge("input_parser", "data_validator")
workflow.add_edge("data_validator", "output_formatter")
workflow.add_edge("output_formatter", END)

# Reasoning: Use for linear workflows where each step must complete before the next
```

### 2. Conditional Edges - Pattern Matching
```python
def pattern_router(state: GraphState) -> str:
    '''Route based on pattern matching in state.'''
    user_input = state.get("user_input", "").lower()
    
    # Pattern-based routing
    if "question" in user_input or "?" in user_input:
        return "question_handler"
    elif "complaint" in user_input or "problem" in user_input:
        return "complaint_handler"
    elif "thank" in user_input or "appreciate" in user_input:
        return "gratitude_handler"
    else:
        return "general_handler"

# Implementation
workflow.add_conditional_edges(
    "intent_analyzer",
    pattern_router,
    {{
        "question_handler": "question_processor",
        "complaint_handler": "complaint_processor", 
        "gratitude_handler": "gratitude_processor",
        "general_handler": "general_processor"
    }}
)
```

### 3. Conditional Edges - Confidence-Based Routing
```python
def confidence_router(state: GraphState) -> str:
    '''Route based on confidence thresholds.'''
    confidence = state.get("confidence_score", 0.0)
    classification = state.get("classification", "unknown")
    
    if confidence >= 0.9:
        return f"high_confidence_{{classification}}"
    elif confidence >= 0.7:
        return f"medium_confidence_{{classification}}"
    elif confidence >= 0.5:
        return "low_confidence_review"
    else:
        return "human_escalation"

workflow.add_conditional_edges(
    "classifier",
    confidence_router,
    {{
        "high_confidence_support": "auto_support",
        "high_confidence_sales": "auto_sales",
        "medium_confidence_support": "assisted_support",
        "medium_confidence_sales": "assisted_sales",
        "low_confidence_review": "confidence_booster",
        "human_escalation": "human_agent",
        "__END__": END
    }}
)
```

### 4. Multi-Condition Complex Routing
```python
def business_logic_router(state: GraphState) -> str:
    '''Complex routing based on multiple business conditions.'''
    user_tier = state.get("user_tier", "standard")
    issue_type = state.get("issue_type", "general")
    urgency = state.get("urgency_level", "normal")
    business_hours = state.get("business_hours", True)
    
    # VIP handling
    if user_tier == "vip":
        if urgency == "critical":
            return "vip_critical_immediate"
        else:
            return "vip_priority_queue"
    
    # After-hours handling
    if not business_hours:
        if urgency == "critical":
            return "after_hours_emergency"
        else:
            return "after_hours_queue"
    
    # Standard business logic
    routing_key = f"{{user_tier}}_{{issue_type}}_{{urgency}}"
    routing_map = {{
        "premium_technical_high": "premium_tech_support",
        "premium_billing_high": "premium_billing_support",
        "standard_technical_normal": "standard_tech_queue",
        "standard_billing_normal": "standard_billing_queue"
    }}
    
    return routing_map.get(routing_key, "general_queue")
```

### 5. Loop-Back Edges with State Tracking
```python
def iteration_controller(state: GraphState) -> str:
    '''Control iterative processing with state tracking.'''
    max_iterations = 5
    current_iteration = state.get("iteration_count", 0)
    success_criteria_met = state.get("success_criteria_met", False)
    improvement_score = state.get("improvement_score", 0.0)
    
    # Success exit condition
    if success_criteria_met:
        return "success_handler"
    
    # Max iterations reached
    if current_iteration >= max_iterations:
        return "max_iterations_handler"
    
    # Improvement stagnation check
    if current_iteration > 2 and improvement_score < 0.1:
        return "stagnation_handler"
    
    # Continue iteration
    return "process_iteration"

workflow.add_conditional_edges(
    "iterative_processor",
    iteration_controller,
    {{
        "success_handler": "finalize_results",
        "max_iterations_handler": "escalate_to_human",
        "stagnation_handler": "alternative_approach",
        "process_iteration": "iterative_processor"  # Loop back
    }}
)
```

### 6. Parallel Processing Edges
```python
# Fan-out pattern - distribute work to multiple parallel nodes
def parallel_work_distributor(state: GraphState) -> List[str]:
    '''Distribute work to multiple parallel processors.'''
    work_items = state.get("work_items", [])
    
    if len(work_items) <= 2:
        return ["single_processor"]
    elif len(work_items) <= 5:
        return ["processor_a", "processor_b"]
    else:
        return ["processor_a", "processor_b", "processor_c"]

# Fan-in pattern - collect results from parallel processors
workflow.add_node("work_distributor", work_distribution_node)
workflow.add_node("processor_a", processor_a_node)
workflow.add_node("processor_b", processor_b_node) 
workflow.add_node("processor_c", processor_c_node)
workflow.add_node("result_collector", result_collection_node)

# Parallel edges
workflow.add_conditional_edges(
    "work_distributor",
    parallel_work_distributor,
    {{
        "single_processor": "processor_a",
        "processor_a": "processor_a",
        "processor_b": "processor_b", 
        "processor_c": "processor_c"
    }}
)

# Convergence edges
workflow.add_edge("processor_a", "result_collector")
workflow.add_edge("processor_b", "result_collector")
workflow.add_edge("processor_c", "result_collector")
```

</EDGE_IMPLEMENTATION_COMPREHENSIVE>

<GRAPH_IMPLEMENTATION_ADVANCED>

## Graph Construction Patterns

### 1. Basic Linear Graph
```python
def create_linear_graph() -> StateGraph:
    '''Simple sequential processing graph.'''
    workflow = StateGraph(GraphState)
    
    # Linear sequence
    workflow.add_node("input", input_node)
    workflow.add_node("process", process_node)
    workflow.add_node("output", output_node)
    
    # Sequential edges
    workflow.add_edge(START, "input")
    workflow.add_edge("input", "process") 
    workflow.add_edge("process", "output")
    workflow.add_edge("output", END)
    
    return workflow

# Reasoning: Use for simple, predictable workflows with no branching
```

### 2. Router-Based Graph Architecture
```python
def create_router_graph() -> StateGraph:
    '''Graph with intelligent routing based on content analysis.'''
    workflow = StateGraph(GraphState)
    
    # Core nodes
    workflow.add_node("analyzer", content_analyzer_node)
    workflow.add_node("router", routing_decision_node)
    
    # Specialized handlers
    workflow.add_node("text_handler", text_processing_node)
    workflow.add_node("image_handler", image_processing_node)
    workflow.add_node("data_handler", data_processing_node)
    workflow.add_node("finalizer", result_finalizer_node)
    
    # Entry and routing
    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "router")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "router", 
        content_type_router,
        {{
            "text": "text_handler",
            "image": "image_handler", 
            "data": "data_handler",
            "unknown": "text_handler"  # Default fallback
        }}
    )
    
    # Convergence
    workflow.add_edge("text_handler", "finalizer")
    workflow.add_edge("image_handler", "finalizer")
    workflow.add_edge("data_handler", "finalizer")
    workflow.add_edge("finalizer", END)
    
    return workflow

# Reasoning: Use for multi-modal or multi-domain processing
```

### 3. Iterative Refinement Graph
```python
def create_iterative_graph() -> StateGraph:
    '''Graph with feedback loops for iterative improvement.'''
    workflow = StateGraph(GraphState)
    
    # Core processing nodes
    workflow.add_node("initializer", initialization_node)
    workflow.add_node("processor", iterative_processor_node)
    workflow.add_node("evaluator", evaluation_node)
    workflow.add_node("refiner", refinement_node)
    workflow.add_node("finalizer", finalization_node)
    
    # Entry point
    workflow.add_edge(START, "initializer")
    workflow.add_edge("initializer", "processor")
    
    # Main processing loop
    workflow.add_edge("processor", "evaluator")
    workflow.add_conditional_edges(
        "evaluator",
        quality_check_router,
        {{
            "needs_refinement": "refiner",
            "acceptable": "finalizer",
            "max_iterations": "finalizer"
        }}
    )
    
    # Refinement loop
    workflow.add_edge("refiner", "processor")  # Loop back
    workflow.add_edge("finalizer", END)
    
    return workflow

# Reasoning: Use for quality-focused tasks requiring iterative improvement
```

### 4. Human-in-the-Loop Graph
```python
def create_hitl_graph() -> StateGraph:
    '''Graph with human oversight and approval mechanisms.'''
    workflow = StateGraph(GraphState)
    
    # Automated processing
    workflow.add_node("preprocessor", preprocessing_node)
    workflow.add_node("ai_processor", ai_processing_node)
    workflow.add_node("quality_check", quality_assessment_node)
    
    # Human interaction nodes
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("revision_handler", revision_handling_node)
    workflow.add_node("approval_handler", approval_handling_node)
    
    # Final processing
    workflow.add_node("postprocessor", postprocessing_node)
    
    # Flow setup
    workflow.add_edge(START, "preprocessor")
    workflow.add_edge("preprocessor", "ai_processor")
    workflow.add_edge("ai_processor", "quality_check")
    
    # Quality-based routing
    workflow.add_conditional_edges(
        "quality_check",
        quality_gate_router,
        {{
            "high_quality": "postprocessor",
            "needs_review": "human_review",
            "auto_reject": "revision_handler"
        }}
    )
    
    # Human review handling
    workflow.add_conditional_edges(
        "human_review",
        human_decision_router,
        {{
            "approved": "approval_handler",
            "needs_revision": "revision_handler",
            "rejected": END
        }}
    )
    
    # Revision loop
    workflow.add_edge("revision_handler", "ai_processor")  # Loop back
    workflow.add_edge("approval_handler", "postprocessor")
    workflow.add_edge("postprocessor", END)
    
    return workflow

# Reasoning: Use for sensitive operations requiring human oversight
```

### 5. Multi-Agent Collaborative Graph
```python
def create_collaborative_graph() -> StateGraph:
    '''Graph with multiple specialized agents working together.'''
    workflow = StateGraph(GraphState)
    
    # Coordination
    workflow.add_node("coordinator", task_coordination_node)
    workflow.add_node("synthesizer", result_synthesis_node)
    
    # Specialized agents
    workflow.add_node("researcher", research_agent_node)
    workflow.add_node("analyzer", analysis_agent_node)
    workflow.add_node("writer", writing_agent_node)
    workflow.add_node("reviewer", review_agent_node)
    
    # Entry and coordination
    workflow.add_edge(START, "coordinator")
    
    # Task distribution
    workflow.add_conditional_edges(
        "coordinator",
        task_distribution_router,
        {{
            "research_needed": "researcher",
            "analysis_needed": "analyzer", 
            "writing_needed": "writer",
            "review_needed": "reviewer",
            "synthesis_ready": "synthesizer"
        }}
    )
    
    # Agent completion routing
    def agent_completion_router(state: GraphState) -> str:
        completed_tasks = state.get("completed_tasks", [])
        required_tasks = state.get("required_tasks", [])
        if set(completed_tasks) >= set(required_tasks):
            return "synthesizer"
        else:
            return "coordinator"  # More work needed
    
    # Agent to coordinator feedback
    workflow.add_conditional_edges("researcher", agent_completion_router, 
                                  {{"coordinator": "coordinator", "synthesizer": "synthesizer"}})
    workflow.add_conditional_edges("analyzer", agent_completion_router,
                                  {{"coordinator": "coordinator", "synthesizer": "synthesizer"}})
    workflow.add_conditional_edges("writer", agent_completion_router,
                                  {{"coordinator": "coordinator", "synthesizer": "synthesizer"}})
    workflow.add_conditional_edges("reviewer", agent_completion_router,
                                  {{"coordinator": "coordinator", "synthesizer": "synthesizer"}})
    
    workflow.add_edge("synthesizer", END)
    
    return workflow

# Reasoning: Use for complex tasks requiring diverse expertise
```

### 6. Graph Compilation and Configuration
```python
def compile_production_graph(workflow: StateGraph) -> CompiledGraph:
    '''Compile graph with production-ready configuration.'''
    
    # Memory configuration
    memory = MemorySaver()
    
    # Compilation with advanced options
    compiled_graph = workflow.compile(
        checkpointer=memory,
        interrupt_before=[],  # Nodes to interrupt before
        interrupt_after=[],   # Nodes to interrupt after
        debug=False,          # Disable in production
    )
    
    return compiled_graph

def compile_development_graph(workflow: StateGraph) -> CompiledGraph:
    '''Compile graph with development-friendly configuration.'''
    
    memory = MemorySaver()
    
    compiled_graph = workflow.compile(
        checkpointer=memory,
        interrupt_before=["human_review", "approval_step"],  # Human interaction points
        interrupt_after=["critical_processor"],              # Debug checkpoints
        debug=True,  # Enable for development
    )
    
    return compiled_graph

# Usage pattern
def create_and_compile_graph() -> CompiledGraph:
    '''Complete graph creation and compilation.'''
    
    # Create graph based on requirements analysis
    if requirements_need_human_oversight():
        workflow = create_hitl_graph()
    elif requirements_need_collaboration():
        workflow = create_collaborative_graph()
    elif requirements_need_iteration():
        workflow = create_iterative_graph()
    else:
        workflow = create_linear_graph()
    
    # Compile based on environment
    if is_production_environment():
        return compile_production_graph(workflow)
    else:
        return compile_development_graph(workflow)
```

</GRAPH_IMPLEMENTATION_ADVANCED>

<IMPLEMENTATION_PATTERNS_2025>

## Pattern 1: LLM with Tool Calling
**When to use:** Node needs to interact with external systems, APIs, databases, or perform specific actions.

**Example Implementation:**
```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def search_customer_database(customer_id: str) -> str:
    '''Search for customer information by ID.'''
    # Direct implementation - no nested LLM calls
    return f"Customer {{customer_id}} data retrieved"

def customer_lookup_node(state: GraphState) -> dict:
    # Reasoning: This node needs external data access, so I use tool calling
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools([search_customer_database])
    
    messages = state["messages"] + [
        ("system", "You are a customer service agent. Use tools to find customer information.")
    ]
    
    response = llm_with_tools.invoke(messages)
    return {{"messages": [response]}}
```

## Pattern 2: LLM with Structured Output
**When to use:** Node needs to make decisions, classify inputs, or extract structured data.

**Example Implementation:**
```python
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal

class IntentClassification(BaseModel):
    '''Structured output for intent classification.'''
    intent: Literal["support", "sales", "billing"] = Field(description="Classified intent")
    confidence: float = Field(description="Confidence score 0-1")
    reasoning: str = Field(description="Brief explanation of classification")

def intent_classifier_node(state: GraphState) -> dict:
    # Reasoning: This node needs structured decision making for routing
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(IntentClassification)
    
    user_message = state["messages"][-1].content
    prompt = f"Classify this user message: {{user_message}}"
    
    result = structured_llm.invoke(prompt)
    return {{
        "messages": [("system", f"Intent classified as: {{result.intent}}")],
        "intent": result.intent,
        "confidence": result.confidence
    }}
```

## Pattern 3: Human-in-the-Loop with Interrupt
**When to use:** Node requires human approval, creative input, or oversight.

**Example Implementation:**
```python
def human_approval_node(state: GraphState) -> dict:
    # Reasoning: This node needs human oversight for quality control
    draft_content = state.get("draft_content", "")
    
    return {{
        "messages": [("system", f"Draft ready for review: {{draft_content}}")],
        "awaiting_approval": True,
        "review_content": draft_content
    }}

# In graph construction:
# graph.add_node("approval", human_approval_node)
# workflow = graph.compile(interrupt_before=["approval"])
```

## Pattern 4: Multi-Step LLM Processing
**When to use:** Complex tasks requiring multiple LLM operations.

**Example Implementation:**
```python
def content_enhancement_node(state: GraphState) -> dict:
    # Reasoning: Multi-step enhancement requires sequential LLM processing
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    raw_content = state.get("raw_content", "")
    
    # Step 1: Structure the content
    structured_prompt = f"Structure this content logically: {{raw_content}}"
    structured = llm.invoke(structured_prompt).content
    
    # Step 2: Enhance with examples
    enhanced_prompt = f"Add relevant examples to: {{structured}}"
    enhanced = llm.invoke(enhanced_prompt).content
    
    return {{
        "messages": [("system", "Content enhanced with structure and examples")],
        "enhanced_content": enhanced,
        "processing_steps": ["structured", "enhanced"]
    }}
```
</IMPLEMENTATION_PATTERNS_2025>

<EDGE_PATTERNS_2025>

## Edge Type 1: Simple Sequential Flow
```python
# Direct progression between nodes
graph.add_edge("node_a", "node_b")
graph.add_edge("node_b", END)
```

## Edge Type 2: Conditional Routing with Structured Output
```python
def route_by_intent(state: GraphState) -> str:
    '''Route based on classified intent from structured output.'''
    intent = state.get("intent", "unknown")
    confidence = state.get("confidence", 0.0)
    
    # Reasoning: High confidence routing to specialized handlers
    if confidence > 0.8:
        return intent  # "support", "sales", or "billing"
    else:
        return "clarification"

# Usage in graph
graph.add_conditional_edges(
    "classifier",
    route_by_intent,
    {{
        "support": "support_handler",
        "sales": "sales_handler", 
        "billing": "billing_handler",
        "clarification": "clarify_intent",
        "__END__": END
    }}
)
```

## Edge Type 3: Multi-Condition Routing
```python
def complex_router(state: GraphState) -> str:
    '''Route based on multiple state conditions.'''
    priority = state.get("priority", "normal")
    user_type = state.get("user_type", "standard")
    
    # Reasoning: Business logic requires different paths for different user types
    if priority == "urgent" and user_type == "premium":
        return "priority_handling"
    elif user_type == "premium":
        return "premium_service"
    elif priority == "urgent":
        return "urgent_handling"
    else:
        return "standard_processing"
```

## Edge Type 4: Loop-Back Edges
```python
def should_continue_processing(state: GraphState) -> str:
    '''Determine if processing loop should continue.'''
    attempts = state.get("processing_attempts", 0)
    success = state.get("processing_success", False)
    
    # Reasoning: Retry logic with maximum attempt limits
    if success:
        return "complete"
    elif attempts < 3:
        return "retry_processing"
    else:
        return "escalate"

graph.add_conditional_edges(
    "processor",
    should_continue_processing,
    {{
        "complete": END,
        "retry_processing": "processor",  # Loop back
        "escalate": "human_handler"
    }}
)
```
</EDGE_PATTERNS_2025>

<COMMON_ARCHITECTURES_2025>

## Foundational Architecture Patterns
*These are starting points for inspiration - feel free to modify, combine, or create entirely new patterns based on your specific requirements.*

### Pattern 1: Linear Processing Pipeline
**Common Use Cases:** Document processing, content transformation, data validation
**Base Structure:** Input → Process → Validate → Output
```python
# Example structure - adapt as needed
graph.add_edge(START, "input_parser")
graph.add_edge("input_parser", "content_processor") 
graph.add_edge("content_processor", "validator")
graph.add_edge("validator", END)
```
**Adaptation Ideas:** Add parallel processing stages, insert quality gates, include human review points

### Pattern 2: Router-Handler Architecture
**Common Use Cases:** Multi-intent systems, content routing, service orchestration
**Base Structure:** Classify → Route → Handle → Respond
```python
# Example structure - customize routing logic
graph.add_edge(START, "intent_classifier")
graph.add_conditional_edges("intent_classifier", custom_router, {{...}})
# Multiple specialized handlers based on your domain
```
**Adaptation Ideas:** Multi-level routing, confidence-based escalation, dynamic handler selection

### Pattern 3: Iterative Refinement System
**Common Use Cases:** Quality improvement, creative processes, optimization tasks
**Base Structure:** Generate → Evaluate → Refine → Repeat
```python
# Example structure - modify convergence criteria
graph.add_edge("generator", "evaluator")
graph.add_conditional_edges("evaluator", refinement_router, {{
    "continue": "refiner",
    "complete": "finalizer"
}})
graph.add_edge("refiner", "generator")  # Loop back
```
**Adaptation Ideas:** Multiple refinement strategies, parallel evaluation, human-guided refinement

### Pattern 4: Human-Supervised Workflow
**Common Use Cases:** Content approval, decision validation, creative collaboration
**Base Structure:** Generate → Review → Approve → Execute
```python
# Example structure - customize approval criteria
graph.add_edge("generator", "human_review")
graph.add_conditional_edges("human_review", approval_router, {{
    "approved": "finalizer",
    "rejected": "generator",
    "escalate": "supervisor_review"
}})
```
**Adaptation Ideas:** Multi-stakeholder approval, conditional automation, escalation paths

## Beyond Standard Patterns: Design Your Own

### Creative Architecture Approaches

#### Domain-Specific Architectures
Design architectures that reflect your specific domain:
- **Medical Workflow**: Symptom Collection → Diagnosis → Treatment Planning → Monitoring
- **Financial Analysis**: Data Gathering → Risk Assessment → Recommendation → Compliance Check
- **Creative Writing**: Ideation → Drafting → Editing → Publication → Feedback Integration
- **Research Process**: Question Formulation → Literature Review → Analysis → Synthesis → Validation

#### Hybrid and Novel Patterns
Combine elements creatively:
- **Adaptive Routing**: Routes change based on learned patterns or user behavior
- **Cascade Processing**: Multiple models of increasing sophistication handle edge cases
- **Consensus Building**: Multiple agents provide input, system synthesizes consensus
- **Learning Loops**: System improves routing/processing based on outcomes
- **Context-Aware Branching**: Architecture adapts based on situational context

#### Complex System Architectures
For sophisticated requirements:
- **Microservice-Style**: Small, focused sub-graphs that communicate
- **Event-Driven**: Reactive processing based on triggers and state changes
- **Hierarchical Processing**: Multiple levels of abstraction and decision-making
- **Federated Learning**: Distributed processing with centralized coordination
- **Self-Organizing**: Dynamic restructuring based on performance metrics

### Architecture Design Questions
Ask yourself these questions to design custom architectures:

1. **Flow Questions:**
   - What is the natural flow of information in this domain?
   - Where are the critical decision points?
   - What are the feedback loops?

2. **Scalability Questions:**
   - How might this system need to grow?
   - What are the bottlenecks?
   - Where might parallel processing help?

3. **Quality Questions:**
   - Where are quality gates needed?
   - What requires human oversight?
   - How do we handle errors and edge cases?

4. **Integration Questions:**
   - What external systems need integration?
   - How do we handle data synchronization?
   - What are the API boundaries?

### Custom Architecture Documentation Template
```python
'''
CUSTOM ARCHITECTURE DESIGN:

Domain Analysis:
- Problem space: [Describe the specific domain and challenges]
- Key stakeholders: [Who interacts with this system?]
- Success criteria: [How do we measure success?]

Architecture Innovation:
- Novel approach: [What makes this architecture unique?]
- Inspiration sources: [What patterns or ideas influenced this design?]
- Design principles: [What principles guided the architecture?]

Implementation Strategy:
- Node distribution: [How are responsibilities divided?]
- Flow control: [How does information and control flow?]
- Error handling: [How do we handle failures and edge cases?]
- Scaling considerations: [How does this architecture scale?]

Trade-offs Made:
- Performance vs. Accuracy: [What trade-offs were considered?]
- Complexity vs. Maintainability: [How was this balanced?]
- Automation vs. Human oversight: [Where is the balance?]
'''
```

## Architectural Flexibility Guidelines

### Do's:
- ✅ Analyze requirements deeply before choosing patterns
- ✅ Combine multiple patterns when beneficial
- ✅ Create domain-specific architectures
- ✅ Innovate based on specific needs
- ✅ Document your architectural reasoning
- ✅ Consider future evolution and scaling

### Don'ts:
- ❌ Force requirements into rigid patterns
- ❌ Assume one-size-fits-all solutions
- ❌ Ignore domain-specific needs
- ❌ Over-complicate simple requirements
- ❌ Under-estimate complex requirements
- ❌ Forget to document design decisions

Remember: The best architecture is the one that **naturally fits your specific requirements**, not the one that looks most impressive or follows the latest trends.

</COMMON_ARCHITECTURES_2025>

<CODE_GENERATION_INSTRUCTIONS>

Generate a single, self-contained, and compilable Python script following this structure:

### 1. Imports and Setup
```python
from typing import Dict, Any, List, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re
import json
```

### 2. State Definition
Always use MessagesState as base and extend as needed:
```python
class GraphState(MessagesState):
    # Add domain-specific fields based on your analysis
    intent: Optional[str] = None
    confidence: Optional[float] = None
    processing_complete: bool = False
    # Add other fields as required by your architecture
```

### 3. Tool Definitions (if needed)
Define tools before node functions that use them:
```python
@tool
def example_tool(param: str) -> str:
    '''Tool description for LLM understanding.'''
    # Direct implementation - avoid nested LLM calls
    return f"Tool result for {{param}}"
```

### 4. Node Implementation
For each node, include reasoning comments:
```python
def node_name(state: GraphState) -> Dict[str, Any]:
    \"\"\"
    Node purpose: [Clear description]
    Implementation reasoning: [Why this pattern was chosen]
    \"\"\"
    # Implementation here
    return {{{{"field": "value"}}
```

### 5. Routing Functions
```python
def router_function(state: GraphState) -> str:
    \"\"\"Route based on state analysis.\"\"\"
    # Routing logic with clear reasoning
    return "next_node_id"
```

## 6. Graph Construction - Flexible and Adaptive
```python
def analyze_and_create_graph(requirements: Dict) -> StateGraph:
    '''
    Analyze requirements and create a custom-tailored graph architecture.
    Don't force requirements into predefined patterns.
    '''
    workflow = StateGraph(GraphState)
    
    # Step 1: Analyze the problem space
    problem_complexity = analyze_complexity(requirements)
    interaction_patterns = identify_interaction_patterns(requirements)
    quality_requirements = extract_quality_needs(requirements)
    
    # Step 2: Design custom architecture based on analysis
    if requires_multi_stage_processing(requirements):
        create_pipeline_with_quality_gates(workflow, requirements)
    
    if requires_dynamic_routing(requirements):
        create_intelligent_routing_system(workflow, requirements)
    
    if requires_human_collaboration(requirements):
        integrate_human_interaction_points(workflow, requirements)
    
    if requires_iterative_improvement(requirements):
        add_feedback_and_refinement_loops(workflow, requirements)
    
    if requires_parallel_processing(requirements):
        implement_concurrent_processing_paths(workflow, requirements)
    
    # Step 3: Add domain-specific customizations
    add_domain_specific_nodes(workflow, requirements)
    implement_custom_routing_logic(workflow, requirements)
    configure_error_handling_strategy(workflow, requirements)
    
    return workflow

def create_adaptive_architecture(json_input: Dict) -> StateGraph:
    '''
    Create architecture that adapts to the specific problem domain.
    This is your primary approach - customize everything.
    '''
    workflow = StateGraph(GraphState)
    
    # Extract and analyze unique requirements
    nodes_needed = analyze_required_nodes(json_input)
    flow_patterns = analyze_required_flows(json_input)
    decision_points = identify_decision_points(json_input)
    
    # Build nodes based on actual requirements, not templates
    for node_spec in nodes_needed:
        node_function = create_custom_node(node_spec)
        workflow.add_node(node_spec['id'], node_function)
    
    # Build edges based on actual flow requirements
    for flow_spec in flow_patterns:
        if flow_spec['type'] == 'conditional':
            router_function = create_custom_router(flow_spec)
            workflow.add_conditional_edges(
                flow_spec['source'],
                router_function,
                flow_spec['targets']
            )
        else:
            workflow.add_edge(flow_spec['source'], flow_spec['target'])
    
    return workflow

# Usage in graph construction section
workflow = analyze_and_create_graph(user_requirements)
# OR
workflow = create_adaptive_architecture(json_dict)
```

### 7. Architecture Reasoning Section
Include a comment block explaining your architectural choices:
```python
\"\"\"
ARCHITECTURE REASONING:
- Chose [pattern] because [requirement] needs [capability]
- Used [tool calling/structured output/interrupts] for [specific nodes] because [reason]
- Selected [LLM model] for [performance/cost/capability] considerations
- Implemented [edge pattern] to handle [business logic/user flow]
\"\"\"
```


</CODE_GENERATION_INSTRUCTIONS>

<QUALITY_CHECKLIST>
Before finalizing your code, verify:
- [ ] All imports are included and correct
- [ ] GraphState properly extends MessagesState  
- [ ] Each node function returns Dict[str, Any]
- [ ] LLM calls include proper error handling
- [ ] Tools are self-contained (no nested LLM calls)
- [ ] Structured output uses proper Pydantic models
- [ ] Conditional edges handle all possible routing outcomes
- [ ] Graph compilation includes appropriate checkpointer
- [ ] Architecture reasoning is clearly documented
- [ ] Code is compilable and logically consistent
</QUALITY_CHECKLIST>

<KEY_EXTRACTION_INSTRUCTIONS>
After generating the complete Python script, add a section titled:

## Required Keys and Credentials

List all environment variables, API keys, and external dependencies needed:
- Environment variables (e.g., OPENAI_API_KEY, GOOGLE_API_KEY)
- Tool-specific credentials 
- External service configurations
- Database connection strings (if applicable)

If no external keys are needed, state: "No external API keys required for this implementation."
</KEY_EXTRACTION_INSTRUCTIONS>

**Final Instructions:**
1. **Custom Architecture First**: Don't force the INPUT into predefined patterns. Analyze the specific requirements and design a tailored solution.
2. **Pattern Inspiration, Not Limitation**: Use the provided patterns as inspiration and building blocks, but feel free to modify, combine, or completely reimagine them.
3. **Domain-Driven Design**: Let the problem domain guide your architecture choices, not architectural fashion or complexity for its own sake.
4. **Reasoning Documentation**: Always explain why you chose your specific architectural approach and how it serves the requirements better than alternatives.
5. **Innovation Encouraged**: Create novel architectures when existing patterns don't fit. The goal is optimal functionality, not pattern conformance.
6. **Scalability Consideration**: Design with future evolution in mind, but don't over-engineer for hypothetical requirements.
7. **User-Centric Design**: Keep the end-user experience and objectives at the center of your architectural decisions.

## Architecture Analysis Questions
Before implementing, ask yourself:
- What makes this problem unique, and how should that influence the architecture?
- What would the ideal flow look like if I weren't constrained by existing patterns?
- How can I combine different approaches to create the best solution for this specific case?
- What are the core requirements that must be satisfied, vs. nice-to-have features?
- How can I make this architecture maintainable and understandable for future developers?

Remember: **The best architecture is the one that naturally emerges from careful analysis of the specific requirements, not the one that follows the most popular patterns.** Your goal is to create intelligent, functional, and maintainable LangGraph applications that excel at their intended purpose.

Now produce the python code based on the inputs provided
""")

def code_node(state: AgentBuilderState):
    """
    LangGraph node to generate the final Python code for the agent.
    It uses the gathered agent_instructions and the CODE_GEN_PROMPT.
    """
    instructions: AgentInstructions = state["agent_instructions"]
 
    # Invoke LLM to generate code based on the detailed prompt and instructions
    code_output = llm.invoke([HumanMessage(content=CODE_GEN_PROMPT.format(
        json_dict=state["json_dict"],
        justification = state["justification"],
        objective=instructions.objective,
        usecases=instructions.usecases,
        examples=instructions.examples
    ))])
    # Return the generated Python code and an AI message
    return {
        "messages": [AIMessage(content="Generated final python code!")],
        "python_code": code_output.content,
    }
