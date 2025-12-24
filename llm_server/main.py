"""Main FastAPI application for the LLM server."""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .config import settings
from .routes import chat, models, completions

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Server",
    description="OpenAI-compatible HTTP wrapper for the llm library",
    version="0.1.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log validation errors with the request body for debugging."""
    body = await request.body()
    if settings.debug:
        logger.debug(f"Validation error for request to {request.url}")
        logger.debug(f"Request body: {body.decode('utf-8', errors='replace')[:2000]}")
        logger.debug(f"Errors: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(body.decode('utf-8', errors='replace')[:500])},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Log all exceptions with full traceback."""
    if settings.debug:
        logger.exception(f"Error handling request to {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": type(exc).__name__}},
    )

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(models.router)
app.include_router(completions.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": settings.model_name or "default"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LLM Server",
        "version": "0.1.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
        },
    }


def run():
    """Run the server using uvicorn."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="LLM Server - OpenAI-compatible wrapper for llm library")
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug logging')
    parser.add_argument('--host', default=None, help=f'Host to bind to (default: {settings.host})')
    parser.add_argument('--port', type=int, default=None, help=f'Port to bind to (default: {settings.port})')
    parser.add_argument('-m', '--model', default=None, help='Default model to use')
    parser.add_argument('-q', '--query', action='append', dest='queries', help='Find model matching query (can be used multiple times)')

    args = parser.parse_args()

    # Update settings from CLI args
    if args.debug:
        settings.debug = True
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.debug("Debug mode enabled")

    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.model:
        settings.model_name = args.model
    if args.queries:
        # Use query-based model selection
        from .routes.chat import find_model_by_query
        model = find_model_by_query(args.queries)
        if model:
            settings.model_name = model.model_id
            if settings.debug:
                logger.debug(f"Selected model via query: {model.model_id}")
        else:
            logger.warning(f"No model found matching queries: {args.queries}")

    uvicorn.run(
        "llm_server.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    run()
