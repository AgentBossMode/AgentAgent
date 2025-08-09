from final_code.states.BaseCopilotRenderingState import BaseCopilotRenderingState
class ReactCopilotState(BaseCopilotRenderingState):
    remaining_steps: int
    structured_response: any
