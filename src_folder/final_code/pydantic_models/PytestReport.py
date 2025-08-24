from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Optional

class TestSummary(BaseModel):
    """Summary of the test run."""
    failed: Optional[int] = None
    passed: Optional[int] = 0
    total: Optional[int] = None
    collected: Optional[int] = None

class TestPhase(BaseModel):
    """Represents a single phase of a test (setup, call, teardown)."""
    duration: Optional[float] = None
    outcome: Optional[str] = None
    
    longrepr: Optional[str] = None

class CrashDetails(BaseModel):
    """Details of a test crash."""
    path: Optional[str] = None
    lineno: Optional[int] = None
    message: Optional[str] = None

class TracebackEntry(BaseModel):
    """A single entry in a traceback."""
    path: Optional[str] = None
    lineno: Optional[int] = None
    message: Optional[str] = None

class CallDetails(BaseModel):
    """Details of the 'call' phase of a test."""
    duration: Optional[float] = None
    outcome: Optional[str] = None
    crash: Optional[CrashDetails] = None
    traceback: Optional[List[TracebackEntry]] = None
    longrepr: Optional[str] = None

class TestResult(BaseModel):
    """Represents the results of a single test case."""
    nodeid: Optional[str] = None
    lineno: Optional[int] = None
    outcome: Optional[str] = None
    keywords: List[str]
    setup: TestPhase
    call: CallDetails
    teardown: TestPhase

class TestWarning(BaseModel):
    """Represents a warning issued during the test run."""
    message: str
    category: str
    when: str
    filename: str
    lineno: int

class PytestReport(BaseModel):
    """Pydantic model for a pytest JSON report."""
    created: Optional[float] = None
    duration: Optional[float] = None
    exitcode: Optional[int] = None
    root: Optional[str] = None
    environment: Dict[str, Any]
    summary: TestSummary
    tests: List[TestResult]
    warnings: List[TestWarning]

