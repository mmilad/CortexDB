"""Command-line launcher for the CortexDB processor sidecar."""

from __future__ import annotations

import argparse
import os

from app.env import load_dotenv

load_dotenv()

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 5010
_HOST_ENV = "CORTEXDB_PROCESSOR_SIDECAR_HOST"
_PORT_ENV = "CORTEXDB_PROCESSOR_SIDECAR_PORT"
_RELOAD_ENV = "CORTEXDB_PROCESSOR_SIDECAR_RELOAD"
_LEGACY_HOST_ENV = "CORTEXDB_PROCESSOR_HOST"
_LEGACY_PORT_ENV = "CORTEXDB_PROCESSOR_PORT"
_LEGACY_RELOAD_ENV = "CORTEXDB_PROCESSOR_RELOAD"


def _env(primary: str, legacy: str, default: str | None = None) -> str | None:
    return os.environ.get(primary, os.environ.get(legacy, default))


def _env_port() -> int:
    raw = _env(_PORT_ENV, _LEGACY_PORT_ENV)
    if raw is None:
        return _DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{_PORT_ENV} must be an integer") from exc
    if port < 1 or port > 65535:
        raise SystemExit(f"{_PORT_ENV} must be between 1 and 65535")
    return port


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CortexDB processor sidecar.")
    parser.add_argument(
        "--host",
        default=_env(_HOST_ENV, _LEGACY_HOST_ENV, _DEFAULT_HOST),
        help=f"Bind host. Can also be set with {_HOST_ENV}.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_port(),
        help=f"Bind port. Can also be set with {_PORT_ENV}. Defaults to 5010.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=(_env(_RELOAD_ENV, _LEGACY_RELOAD_ENV, "") or "").lower() in ("1", "true", "yes"),
        help=f"Enable uvicorn auto-reload. Can also be set with {_RELOAD_ENV}=true.",
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("app.processors.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
