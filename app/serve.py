"""Command-line launcher for the CortexDB FastAPI server."""

from __future__ import annotations

import argparse
import os

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 5000


def _env_port() -> int:
    raw = os.environ.get("CORTEXDB_API_PORT")
    if raw is None:
        return _DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError as exc:
        raise SystemExit("CORTEXDB_API_PORT must be an integer") from exc
    if port < 1 or port > 65535:
        raise SystemExit("CORTEXDB_API_PORT must be between 1 and 65535")
    return port


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CortexDB API server.")
    parser.add_argument(
        "--host",
        default=os.environ.get("CORTEXDB_API_HOST", _DEFAULT_HOST),
        help="Bind host. Can also be set with CORTEXDB_API_HOST.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_port(),
        help="Bind port. Can also be set with CORTEXDB_API_PORT. Defaults to 5000.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.environ.get("CORTEXDB_API_RELOAD", "").lower() in ("1", "true", "yes"),
        help="Enable uvicorn auto-reload. Can also be set with CORTEXDB_API_RELOAD=true.",
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
