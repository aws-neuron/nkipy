import os
import re
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py


def fix_proto_imports(proto_dir: str) -> None:
    """Fix the generated protobuf imports to use absolute package paths.

    The protoc compiler generates imports like:
        from xla import xla_data_pb2
        from xla.service import metrics_pb2

    We need to convert these to absolute imports:
        from nkipy.third_party.xla import xla_data_pb2
        from nkipy.third_party.xla.service import metrics_pb2
    """
    # Files to fix and their locations
    files_to_fix = [
        os.path.join(proto_dir, "xla", "service", "hlo_pb2.py"),
        os.path.join(proto_dir, "xla", "service", "metrics_pb2.py"),
        os.path.join(proto_dir, "xla", "xla_data_pb2.py"),
    ]

    for filepath in files_to_fix:
        if not os.path.exists(filepath):
            continue

        with open(filepath, "r") as f:
            content = f.read()

        # Fix imports from xla.service -> nkipy.third_party.xla.service
        content = re.sub(
            r"from xla\.service import",
            "from nkipy.third_party.xla.service import",
            content,
        )

        # Fix imports from xla import -> nkipy.third_party.xla import
        content = re.sub(
            r"from xla import",
            "from nkipy.third_party.xla import",
            content,
        )

        with open(filepath, "w") as f:
            f.write(content)


class BuildProtos(build_py):
    """Custom build command to compile proto files before building."""

    def run(self):
        # Proto files are in third_party
        proto_dir = os.path.join(os.path.dirname(__file__), "src/nkipy/third_party")

        # Proto files to compile (order matters - dependencies first)
        proto_files = [
            "xla/xla_data.proto",
            "xla/service/metrics.proto",
            "xla/service/hlo.proto",
        ]

        for proto_file in proto_files:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "grpc_tools.protoc",
                    f"-I{proto_dir}",
                    f"--python_out={proto_dir}",
                    os.path.join(proto_dir, proto_file),
                ]
            )

        # Fix the generated imports to use absolute package paths
        fix_proto_imports(proto_dir)

        super().run()


setup(cmdclass={"build_py": BuildProtos})
