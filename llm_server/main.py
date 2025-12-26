"""Main FastAPI application for the LLM server."""

import atexit
import errno
import logging
import os
import sys

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
            "models": "/v1/models",
            "health": "/health",
        },
    }


def _remove_pidfile(pidfile: str):
    """Remove the PID file on exit."""
    try:
        os.remove(pidfile)
    except OSError:
        pass


def _check_pidfile(pidfile: str) -> int | None:
    """Check if a PID file exists and if the process is running. Returns PID if running."""
    try:
        with open(pidfile, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # Check if process exists
        return pid
    except (FileNotFoundError, ValueError):
        return None
    except OSError as e:
        if e.errno == errno.ESRCH:  # No such process
            return None
        if e.errno == errno.EPERM:  # Process exists but not owned by us
            return pid
        return None


def _daemonize(pidfile: str | None, logfile: str | None):
    """Daemonize the current process using double-fork."""
    if sys.platform == "win32":
        sys.stderr.write("Error: --daemon is not supported on Windows\n")
        sys.exit(1)

    # Check for already running daemon
    if pidfile:
        existing_pid = _check_pidfile(pidfile)
        if existing_pid:
            sys.stderr.write(f"Error: Daemon already running with PID {existing_pid}\n")
            sys.exit(1)

    # First fork
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f"First fork failed: {e}\n")
        sys.exit(1)

    # Decouple from parent environment
    os.chdir("/")
    os.setsid()
    os.umask(0)

    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f"Second fork failed: {e}\n")
        sys.exit(1)

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()

    # Open and duplicate file descriptors, then close the originals
    devnull = open(os.devnull, "r+b")
    os.dup2(devnull.fileno(), sys.stdin.fileno())

    if logfile:
        # Ensure log directory exists
        logfile_dir = os.path.dirname(logfile)
        if logfile_dir and not os.path.exists(logfile_dir):
            os.makedirs(logfile_dir, exist_ok=True)
        log_fd = open(logfile, "a+")
        os.dup2(log_fd.fileno(), sys.stdout.fileno())
        os.dup2(log_fd.fileno(), sys.stderr.fileno())
        log_fd.close()  # Close original after dup2
    else:
        os.dup2(devnull.fileno(), sys.stdout.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())

    devnull.close()  # Close original after dup2

    # Write PID file
    if pidfile:
        # Ensure PID file directory exists
        pidfile_dir = os.path.dirname(pidfile)
        if pidfile_dir and not os.path.exists(pidfile_dir):
            os.makedirs(pidfile_dir, exist_ok=True)
        with open(pidfile, "w") as f:
            f.write(f"{os.getpid()}\n")
        atexit.register(_remove_pidfile, pidfile)


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
        _daemonize(settings.pidfile, settings.logfile)

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
