import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        raw = os.getenv("SCALAR_API_KEYS", "")
        self.valid_keys = {k.strip() for k in raw.split(",") if k.strip()}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        key = request.headers.get("X-API-Key", "")
        if key not in self.valid_keys:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Pass X-API-Key header."}
            )
        return await call_next(request)
