multi_pattern = """
**When to use:** Complex tasks requiring multiple LLM operations.

**Example Implementation:**
```python
def content_enhancement_node(state: GraphState) -> dict:
    # Reasoning: Multi-step enhancement requires sequential LLM processing    
    raw_content = state.get("raw_content", "")
    
    # Step 1: Structure the content
    structured_prompt = f"Structure this content logically: {{raw_content}}"
    structured = llm.invoke(structured_prompt).content
    
    # Step 2: Enhance with examples
    enhanced_prompt = f"Add relevant examples to: {{structured}}"
    enhanced = llm.invoke(enhanced_prompt).content
    
    return {{
        "messages": [AIMessage(content= "Content enhanced with structure and examples")],
        "enhanced_content": enhanced,
        "processing_steps": ["structured", "enhanced"]
    }}
```
"""