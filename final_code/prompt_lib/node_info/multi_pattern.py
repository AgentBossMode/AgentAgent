multi_pattern = """
**When to use:** Tasks requiring multiple sequential LLM operations within a single node, where each step builds on previous outputs.

**Basic Node Structure:**
```python
def multi_step_node(state: GraphState) -> dict:
    input_data = state.get("input_key", "")
    
    # Step 1: First operation
    result_a = llm.invoke(f"Operation A on: {{input_data}}").content
    
    # Step 2: Build on step 1
    result_b = llm.invoke(f"Operation B using: {{result_a}}").content
    
    # Step 3: Final processing
    final = llm.invoke(f"Operation C combining: {{result_a}} and {{result_b}}").content
    
    return {
        "messages": [AIMessage(content="Processing complete")],
        "output": final,
        "intermediates": {{"step1": result_a, "step2": result_b}}
    }
```

**Common Patterns:**

**Sequential Processing:**
*Use when:* Tasks need progressive refinement or building complexity
*Examples:* Recipe development (ingredients → method → presentation), legal brief writing (facts → arguments → conclusion), product design (concept → prototype → specifications)
```python
# Each step enhances previous result
step1 = llm.invoke(f"Process: {{input}}").content
step2 = llm.invoke(f"Enhance: {{step1}}").content
final = llm.invoke(f"Finalize: {{step2}}").content
```

**Multi-Aspect Analysis:**
*Use when:* Need to examine different dimensions of the same input
*Examples:* Market research (demographics + competitors + trends), medical diagnosis (symptoms + history + tests), financial assessment (revenue + costs + risks)
```python
# Analyze different aspects, then combine
aspect_x = llm.invoke(f"Analyze X in: {{data}}").content
aspect_y = llm.invoke(f"Analyze Y in: {{data}}").content
combined = llm.invoke(f"Combine: {{aspect_x}} and {{aspect_y}}").content
```

**Validation Loop:**
*Use when:* Output quality varies and needs iterative improvement
*Examples:* Translation refinement (translate → check accuracy → improve), customer service responses (draft → tone check → personalize), mathematical proofs (solve → verify logic → correct errors)
```python
# Generate then validate/improve
output = llm.with_structured_output(OutputSchema).invoke(f"Generate: {{input}}")
validation = llm.with_structured_output(ValidationSchema).invoke(f"Validate: {{output}}")
if validation.needs_improvement:
    output = llm.with_structured_output(OutputSchema).invoke(f"Improve: {{output}} based on: {{validation}}")
```

**Key Guidelines:**
- Use descriptive prompts that clearly define each step's purpose
- Return structured state that downstream nodes can consume
- Consider token limits when chaining operations
- Pass relevant context from state to each prompt
"""