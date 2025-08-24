from .assertion_failures_debugging_guide import assertion_failures_debugging_guide
from .runtime_failures_debugging_guide import runtime_failures_debugging_guide
from .syntax_failures_debugging_guide import syntax_failures_debugging_guide

debugging_tsg = f"""
<ASSERTION_FAILURE_TSG>
{assertion_failures_debugging_guide}
</ASSERTION_FAILURE_TSG>

<RUNTIME_FAILURE_TSG>
{runtime_failures_debugging_guide}
</RUNTIME_FAILURE_TSG>

<SYNTAX_FAILURE_TSG>
{syntax_failures_debugging_guide}
</SYNTAX_FAILURE_TSG>
"""