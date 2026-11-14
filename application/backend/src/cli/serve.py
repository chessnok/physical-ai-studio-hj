"""Serve backend and frontend CLI commands."""

import os
from pathlib import Path

import click

from cli.database import _run_migrations
from core.security import get_ssh_feature_availability
from settings import get_settings

GRACEFUL_SHUTDOWN_S = 10
"""How long open connections get to finish when the server is asked to stop."""


def _package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _configure_packaged_runtime() -> None:
    package_root = _package_root()
    bundled_webui = package_root / "webui"

    if bundled_webui.joinpath("index.html").exists():
        os.environ.setdefault("STATIC_FILES_DIR", str(bundled_webui))

    os.environ.setdefault("ALEMBIC_CONFIG_PATH", str(package_root / "alembic.ini"))
    os.environ.setdefault("ALEMBIC_SCRIPT_LOCATION", str(package_root / "alembic"))

    # Settings are resolved for each call, so downstream modules observe these paths.


def start_server(host: str, port: int) -> None:
    """Configure the packaged runtime, run migrations, and launch the API server."""
    _configure_packaged_runtime()
    # `--host`/`--port` can override the settings-derived default below; keep
    # `Settings.host` in agreement with the address actually handed to uvicorn,
    # so anything that reasons about the real bind address (the SSH feature's
    # loopback-only enforcement) never checks a stale default instead of it.
    os.environ["HOST"] = host
    os.environ["PORT"] = str(port)
    get_ssh_feature_availability.cache_clear()
    _run_migrations()

    import uvicorn

    from utils.multiprocessing import ensure_spawn_start_method

    ensure_spawn_start_method()
    # Without a timeout uvicorn waits for every open connection's task on SIGTERM, forever if one
    # never ends, and only a second Ctrl+C forces it: SIGTERM looked ignored, and a SIGKILL then
    # left camera publishers behind. After the timeout the tasks are cancelled and the
    # application still shuts down: sessions close and camera publishers are stopped.
    uvicorn.run("main:app", host=host, port=port, timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S)


@click.command()
@click.option(
    "--host",
    default=lambda: get_settings().host,
    show_default=True,
    help="Bind address (HOST setting). Keep loopback unless network exposure is intended.",
)
@click.option(
    "--port", default=lambda: get_settings().port, show_default=True, type=int, help="HTTP port (PORT setting)."
)
def serve(host: str, port: int) -> None:
    """Run migrations and start the Studio web application.

    Uses the configured local storage and database. Startup can modify them.
    Studio's API has no authentication model; do not expose it to untrusted networks.

    \b
    Example:
      physicalai-studio serve --host 127.0.0.1 --port 7861
    """
    start_server(host, port)
