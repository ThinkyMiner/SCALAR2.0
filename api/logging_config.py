import json
import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs each request as a JSON line: method, path, status, latency_ms."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logging.info(json.dumps({
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": elapsed_ms,
        }))
        return response
