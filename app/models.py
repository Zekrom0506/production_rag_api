from pydantic import BaseModel, Field
from datetime import datetime, timezone


class ChatRequest(BaseModel):
    """Incoming chat request"""
    message: str = Field( 
        ..., #means this field is required
        min_length=1,  #usually very small and very large messages are suspicious
        max_length=10000,  # Limit message length to prevent abuse before reaching the LLM
        description="The user's message to the agent" #metadata for documentation purposes
    )
    thread_id: str = Field(
        default="default",
        description="Conversation thread ID"
    )


class ChatResponse(BaseModel):
    """Chat response returned to client"""
    response: str
    thread_id: str
    model_used: str
    cached: bool = False
    processing_time_ms: float
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    environment: str
    version: str = "1.0.0"
    checks: dict = {}


class MetricsResponse(BaseModel):
    """Metrics endpoint response"""
    total_requests: int 
    total_errors: int
    error_rate: str
    average_latency_ms: float
    cache_hit_rate: str
    total_input_tokens: int
    total_output_tokens: int


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: str | None = None #type can be str or None , if not provided it will be None
    request_id: str | None = None 