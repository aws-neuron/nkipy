"""CLI entrypoint to launch the nkipy HTTP server."""

from nkipy_serving.entrypoints import http_server
from nkipy_serving.server_args import ServerArgs


def main(argv: list[str] | None = None) -> None:
    server_args = ServerArgs.from_cli(argv)
    http_server.launch(server_args)


if __name__ == "__main__":
    main()
