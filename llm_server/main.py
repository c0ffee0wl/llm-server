"""Main FastAPI application for the LLM server."""

import logging
import os
import sys

import daemon
from daemon import pidfile as daemon_pidfile
from lockfile import AlreadyLocked, LockTimeout

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
        content={"error": {"message": str(exc), "type": type(exc).__name__, "code": "internal_error"}},
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
            "completions": "/v1/completions",
            "models": "/v1/models",
            "health": "/health",
        },
    }


def _run_as_daemon(pidfile_path: str | None, logfile: str | None):
    """Run the server as a daemon using python-daemon (PEP 3143)."""
    if sys.platform == "win32":
        sys.stderr.write("Error: --daemon is not supported on Windows\n")
        sys.exit(1)

    # Ensure directories exist before opening files
    if pidfile_path:
        pidfile_dir = os.path.dirname(pidfile_path)
        if pidfile_dir and not os.path.exists(pidfile_dir):
            os.makedirs(pidfile_dir, exist_ok=True)
    if logfile:
        logfile_dir = os.path.dirname(logfile)
        if logfile_dir and not os.path.exists(logfile_dir):
            os.makedirs(logfile_dir, exist_ok=True)

    # Prepare file handles (must be opened before daemonizing)
    stdout_file = open(logfile, 'a+') if logfile else None
    stderr_file = stdout_file  # Share the same file handle

    # Create pidfile lock (TimeoutPIDLockFile with acquire_timeout=0 for immediate fail)
    pidfile_lock = daemon_pidfile.TimeoutPIDLockFile(pidfile_path, acquire_timeout=0) if pidfile_path else None

    context = daemon.DaemonContext(
        working_directory='/',
        umask=0,
        pidfile=pidfile_lock,
        stdout=stdout_file,
        stderr=stderr_file,
    )

    try:
        context.open()
    except (AlreadyLocked, LockTimeout):
        sys.stderr.write(f"Error: Daemon already running (pidfile locked: {pidfile_path})\n")
        sys.exit(1)
    # Note: Don't close context - daemon runs until terminated
    # atexit handler is registered automatically by DaemonContext.open()


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
    parser.add_argument('-n', '--no-log', action='store_true', dest='no_log', help="Don't log requests/responses to database")
    parser.add_argument('--pidfile', default=None, help='PID file path (used with --daemon)')
    parser.add_argument('--logfile', default=None, help='Log file path (used with --daemon)')

    # Mutually exclusive daemon/service options
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--daemon', action='store_true', help='Run as a background daemon')
    mode_group.add_argument('--service', action='store_true',
                           help='Install and start as a systemd socket-activated service')
    mode_group.add_argument('--uninstall-service', action='store_true',
                           help='Uninstall the systemd service')

    # Service-specific options
    parser.add_argument('--system', action='store_true',
                       help='Install as system-level service (requires root, default is user-level)')

    args = parser.parse_args()

    # -q and --service are mutually exclusive (use -m with --service instead)
    if args.queries and args.service:
        parser.error("-q/--query cannot be used with --service. Use -m/--model instead.")

    # Handle --service flag (install systemd service)
    if args.service:
        from .systemd_service import install_service
        success = install_service(
            host=args.host or settings.host,
            port=args.port or settings.port,
            model=args.model,
            user_mode=not args.system,
            debug=args.debug,
            no_log=args.no_log,
        )
        sys.exit(0 if success else 1)

    # Handle --uninstall-service flag
    if args.uninstall_service:
        from .systemd_service import uninstall_service
        success = uninstall_service(user_mode=not args.system)
        sys.exit(0 if success else 1)

    # Update settings from CLI args (before daemonization)
    if args.debug:
        settings.debug = True
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.model:
        settings.model_name = args.model
    if args.pidfile:
        settings.pidfile = args.pidfile
    if args.logfile:
        settings.logfile = args.logfile
    if args.no_log:
        settings.no_log = True

    # Daemonize before configuring logging (redirects file descriptors)
    if args.daemon:
        _run_as_daemon(settings.pidfile, settings.logfile)

    # Configure logging after daemonization so handlers use correct fds
    if settings.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.debug("Debug mode enabled")

    if args.queries:
        # Use query-based model selection
        from .config import find_model_by_query
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
        reload=settings.debug and not args.daemon,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    run()
