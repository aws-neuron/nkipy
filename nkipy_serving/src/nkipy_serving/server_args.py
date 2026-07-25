"""CLI/runtime args for the nkipy HTTP server launcher."""

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerArgs:
    host: str = "127.0.0.1"
    port: int = 30000
    log_level: str = "info"
    log_level_http: str | None = None
    config_path: str | None = None
    max_model_len: int | None = None
    device_offset: int | None = None
    workers: int = 1
    access_log: bool = False

    def check_server_args(self) -> None:
        if not self.host.strip():
            raise RuntimeError("host must be non-empty")
        if self.port <= 0 or self.port > 65535:
            raise RuntimeError(f"port must be in [1, 65535], got {self.port}")
        if self.device_offset is not None and self.device_offset < 0:
            raise RuntimeError(
                f"device_offset must be >= 0 when set, got {self.device_offset}"
            )
        if self.workers != 1:
            raise RuntimeError(
                "Bootstrap runtime only supports a single uvicorn worker. "
                f"Got workers={self.workers}"
            )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--host",
            type=str,
            default=cls.host,
            help="Host for the HTTP server.",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=cls.port,
            help="Port for the HTTP server.",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default=cls.log_level,
            help="Default log level.",
        )
        parser.add_argument(
            "--log-level-http",
            type=str,
            default=cls.log_level_http,
            help="HTTP server log level; defaults to --log-level when unset.",
        )
        parser.add_argument(
            "--config",
            type=str,
            default=cls.config_path,
            help="Path to runtime JSON config file.",
        )
        parser.add_argument(
            "--max-model-len",
            type=int,
            default=cls.max_model_len,
            help="Maximum context/sequence length. Overrides config file.",
        )
        parser.add_argument(
            "--device-offset",
            type=int,
            default=cls.device_offset,
            help="Base Neuron core index for worker placement. Overrides config file.",
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=cls.workers,
            help="Number of uvicorn workers. Prototype supports only 1.",
        )
        parser.add_argument(
            "--access-log",
            action="store_true",
            help="Enable uvicorn access log.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "ServerArgs":
        return cls(
            host=str(args.host),
            port=int(args.port),
            log_level=str(args.log_level),
            log_level_http=(
                str(args.log_level_http) if args.log_level_http is not None else None
            ),
            config_path=str(args.config) if args.config is not None else None,
            max_model_len=int(args.max_model_len)
            if args.max_model_len is not None
            else None,
            device_offset=int(args.device_offset)
            if args.device_offset is not None
            else None,
            workers=int(args.workers),
            access_log=bool(args.access_log),
        )

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "ServerArgs":
        parser = argparse.ArgumentParser()
        cls.add_cli_args(parser)
        return cls.from_cli_args(parser.parse_args(argv))
