"""JSON-line protocol models for media worker subprocess communication.

Defines the request/response envelope structures and worker lifecycle states.
"""

from __future__ import annotations

import secrets
from enum import Enum
from typing import Any

from pydantic import BaseModel, model_validator


class WorkerStatus(str, Enum):
    """Lifecycle states of a media worker process."""

    IDLE = "idle"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


class WorkerRequest(BaseModel):
    """Request envelope — sent to worker subprocess via stdin."""

    id: str = ""
    method: str
    params: dict[str, Any] = {}
    timeout: int | None = None

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = secrets.token_hex(6)


class ErrorDetail(BaseModel):
    """Structured error detail within a WorkerResponse."""

    code: str
    message: str


class WorkerResponse(BaseModel):
    """Response envelope — received from worker subprocess via stdout."""

    id: str
    result: dict[str, Any] | None = None
    error: ErrorDetail | None = None
    metadata: dict[str, Any] = {}

    @model_validator(mode="after")
    def _check_result_xor_error(self) -> WorkerResponse:
        has_result = self.result is not None
        has_error = self.error is not None
        if has_result == has_error:
            raise ValueError("exactly one of 'result' or 'error' must be set")
        return self
