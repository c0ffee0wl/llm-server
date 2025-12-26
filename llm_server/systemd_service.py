"""Systemd service and socket unit management for llm-server."""

import os
import sys
from pathlib import Path
from typing import Optional


# Unit file templates
SOCKET_UNIT_TEMPLATE = """\
[Unit]
Description=LLM Server Socket
Documentation=https://github.com/your-repo/llm-server

[Socket]
ListenStream={listen_address}
Accept=no
ReusePort=true

[Install]
WantedBy=sockets.target
"""

SERVICE_UNIT_TEMPLATE = """\
[Unit]
Description=LLM Server - OpenAI-compatible wrapper for llm library
Documentation=https://github.com/your-repo/llm-server
Requires={socket_unit}
After=network.target {socket_unit}

[Service]
Type=simple
# Use --fd 3 for socket activation (FD 3 is SD_LISTEN_FDS_START)
ExecStart={executable} {extra_args}
Restart=on-failure
RestartSec=5
{environment}

[Install]
WantedBy=multi-user.target
"""


def get_unit_directory(user_mode: bool = True) -> Path:
    """
    Get the appropriate systemd unit directory.

    Args:
        user_mode: If True, use user-level units (~/.config/systemd/user/)
                   If False, use system-level units (/etc/systemd/system/)
    """
    if user_mode:
        config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        return config_home / "systemd" / "user"
    else:
        return Path("/etc/systemd/system")


def get_service_name() -> str:
    """Get the base service name."""
    return "llm-server"


def generate_socket_unit(host: str, port: int) -> str:
    """Generate the .socket unit file content."""
    # For socket activation, use host:port format
    if host == "0.0.0.0" or host == "::":
        listen_address = str(port)  # Listen on all interfaces
    else:
        listen_address = f"{host}:{port}"
    return SOCKET_UNIT_TEMPLATE.format(listen_address=listen_address)


def generate_service_unit(
    socket_unit: str,
    model: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8777,
    debug: bool = False,
    no_log: bool = False,
) -> str:
    """Generate the .service unit file content."""
    # Find the llm-server executable - use the python from this virtualenv
    python_executable = sys.executable

    # For socket activation, use uvicorn directly with --fd 3
    # FD 3 is SD_LISTEN_FDS_START (the socket passed by systemd)
    executable = f"{python_executable} -m uvicorn llm_server.main:app --fd 3"

    # Build extra arguments (no --host/--port since we use socket activation)
    extra_args = []
    if debug:
        extra_args.append("--log-level debug")
    else:
        extra_args.append("--log-level info")

    # Environment variables
    env_lines = []
    # Use explicit model arg, fall back to current env var
    model_name = model or os.environ.get("LLM_SERVER_MODEL_NAME")
    if model_name:
        env_lines.append(f"Environment=LLM_SERVER_MODEL_NAME={model_name}")
    if no_log:
        env_lines.append("Environment=LLM_SERVER_NO_LOG=true")
    if debug:
        env_lines.append("Environment=LLM_SERVER_DEBUG=true")

    return SERVICE_UNIT_TEMPLATE.format(
        socket_unit=socket_unit,
        executable=executable,
        extra_args=" ".join(extra_args),
        environment="\n".join(env_lines),
    )


def install_service(
    host: str = "127.0.0.1",
    port: int = 8777,
    model: Optional[str] = None,
    user_mode: bool = True,
    debug: bool = False,
    no_log: bool = False,
) -> bool:
    """
    Install and enable the systemd socket and service units.

    Returns True on success, False on failure.
    """
    try:
        from pystemd.systemd1 import Manager
        from pystemd.dbuslib import DBus
    except ImportError:
        print("Error: pystemd is required for --service. Install with:")
        print("  pip install 'llm-server[service]'")
        return False

    service_name = get_service_name()
    socket_unit_name = f"{service_name}.socket"
    service_unit_name = f"{service_name}.service"

    unit_dir = get_unit_directory(user_mode)
    unit_dir.mkdir(parents=True, exist_ok=True)

    socket_path = unit_dir / socket_unit_name
    service_path = unit_dir / service_unit_name

    # Generate unit files
    socket_content = generate_socket_unit(host, port)
    service_content = generate_service_unit(
        socket_unit=socket_unit_name,
        model=model,
        host=host,
        port=port,
        debug=debug,
        no_log=no_log,
    )

    # Write unit files
    print(f"Writing {socket_path}")
    socket_path.write_text(socket_content)

    print(f"Writing {service_path}")
    service_path.write_text(service_content)

    # Reload systemd and enable/start units
    try:
        if user_mode:
            bus = DBus(user_mode=True)
        else:
            bus = DBus()

        bus.open()
        try:
            manager = Manager(bus=bus)
            manager.load()

            # Reload daemon to pick up new unit files
            print("Reloading systemd daemon...")
            manager.Manager.Reload()

            # Enable and start the socket
            print(f"Enabling {socket_unit_name}...")
            manager.Manager.EnableUnitFiles([socket_unit_name.encode()], False, True)

            print(f"Starting {socket_unit_name}...")
            manager.Manager.StartUnit(socket_unit_name.encode(), b"replace")

            print(f"\nService installed successfully!")
            print(f"Socket: {socket_path}")
            print(f"Service: {service_path}")
            print(f"\nThe service will start automatically when connections arrive on port {port}")
            print(f"\nTo check status:")
            if user_mode:
                print(f"  systemctl --user status {socket_unit_name}")
            else:
                print(f"  systemctl status {socket_unit_name}")

            return True
        finally:
            bus.close()

    except Exception as e:
        print(f"Error enabling/starting service: {e}")
        print("\nUnit files were written successfully. You can try manually with:")
        if user_mode:
            print(f"  systemctl --user daemon-reload")
            print(f"  systemctl --user enable --now {socket_unit_name}")
        else:
            print(f"  sudo systemctl daemon-reload")
            print(f"  sudo systemctl enable --now {socket_unit_name}")
        return False


def uninstall_service(user_mode: bool = True) -> bool:
    """
    Stop, disable, and remove the systemd units.

    Returns True on success, False on failure.
    """
    try:
        from pystemd.systemd1 import Manager
        from pystemd.dbuslib import DBus
    except ImportError:
        print("Error: pystemd is required. Install with:")
        print("  pip install 'llm-server[service]'")
        return False

    service_name = get_service_name()
    socket_unit_name = f"{service_name}.socket"
    service_unit_name = f"{service_name}.service"

    unit_dir = get_unit_directory(user_mode)
    socket_path = unit_dir / socket_unit_name
    service_path = unit_dir / service_unit_name

    try:
        if user_mode:
            bus = DBus(user_mode=True)
        else:
            bus = DBus()

        bus.open()
        try:
            manager = Manager(bus=bus)
            manager.load()

            # Stop units
            print(f"Stopping {socket_unit_name}...")
            try:
                manager.Manager.StopUnit(socket_unit_name.encode(), b"replace")
            except Exception:
                pass  # May not be running

            print(f"Stopping {service_unit_name}...")
            try:
                manager.Manager.StopUnit(service_unit_name.encode(), b"replace")
            except Exception:
                pass

            # Disable units
            print(f"Disabling units...")
            try:
                manager.Manager.DisableUnitFiles([socket_unit_name.encode()], False)
            except Exception:
                pass

            # Remove unit files
            if socket_path.exists():
                print(f"Removing {socket_path}")
                socket_path.unlink()

            if service_path.exists():
                print(f"Removing {service_path}")
                service_path.unlink()

            # Reload daemon
            manager.Manager.Reload()

            print("\nService uninstalled successfully!")
            return True
        finally:
            bus.close()

    except Exception as e:
        print(f"Error uninstalling service: {e}")
        # Try to at least remove the files
        if socket_path.exists():
            print(f"Removing {socket_path}")
            socket_path.unlink()
        if service_path.exists():
            print(f"Removing {service_path}")
            service_path.unlink()
        print("\nUnit files removed. You may need to run:")
        if user_mode:
            print(f"  systemctl --user daemon-reload")
        else:
            print(f"  sudo systemctl daemon-reload")
        return False
