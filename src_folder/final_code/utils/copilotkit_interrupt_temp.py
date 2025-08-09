from typing import Optional
import uuid
from typing import Dict, Any
from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from copilotkit.langgraph import copilotkit_messages_to_langchain

def copilotkit_interrupt(
        message: Optional[str] = None,
        action: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None
):
    if message is None and action is None:
        raise ValueError('Either message or action (and optional arguments) must be provided')

    interrupt_message = None
    interrupt_values = None
    answer = None

    if message is not None:
        interrupt_values = message
        interrupt_message = AIMessage(content=message, id=str(uuid.uuid4()))
    else:
        tool_id = str(uuid.uuid4())
        interrupt_message = AIMessage(
                content="",
                tool_calls=[{
                    "id": tool_id,
                    "name": action,
                    "args": args or {}
                }]
            )
        interrupt_values = {
            "action": action,
            "args": args or {}
        }

    response = interrupt({
        "__copilotkit_interrupt_value__": interrupt_values,
        "__copilotkit_messages__": [interrupt_message]
    })

    if isinstance(response, str) or isinstance(response, dict):
        return response, [interrupt_message]

    handler = copilotkit_messages_to_langchain()
    response = handler(response)
    answer = response[-1].content

    return answer, response

