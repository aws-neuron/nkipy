import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools import find_packages

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        # Clean stale build artifacts before building
        import shutil
        stale_dirs = ["nkipy_kernelgen/_mlir"]
        for dir_path in stale_dirs:
            if os.path.exists(dir_path):
                print(f"Removing stale artifacts: {dir_path}")
                shutil.rmtree(dir_path)

        # Ensure CMake is installed
        try:
            subprocess.check_call(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        # Call the build process for each extension
        for ext in self.extensions:
            self.build_extension(ext)

        # After building, copy the _mlir bindings to nkipy package
        self.copy_mlir_bindings()

    def build_extension(self, ext):
        # Retrieve LLVM_BUILD_DIR from environment variable (optional)
        llvm_build_dir = os.environ.get("LLVM_BUILD_DIR")

        cmake_args = [
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPython_EXECUTABLE={sys.executable}",
        ]

        # Only set MLIR_DIR if LLVM_BUILD_DIR is provided
        # Otherwise, let CMake auto-detect using find_package
        if llvm_build_dir:
            cmake_args += [f"-DMLIR_DIR={llvm_build_dir}/lib/cmake/mlir"]

        build_temp = os.path.abspath("build")
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        BUILD_WITH = os.environ.get("BUILD_WITH")
        if not BUILD_WITH or BUILD_WITH == "ninja":
            subprocess.run(
                ["cmake", "-G Ninja", ext.sourcedir] + cmake_args,
                cwd=build_temp,
                check=True,
            )
            if NUM_THREADS := os.environ.get("NUM_THREADS"):
                subprocess.run(
                    ["ninja", f"-j{NUM_THREADS}"], cwd=build_temp, check=True
                )
            else:
                subprocess.run(["ninja"], cwd=build_temp, check=True)
        elif BUILD_WITH == "make":
            subprocess.run(
                ["cmake", "-G Unix Makefiles", ext.sourcedir] + cmake_args,
                cwd=build_temp,
                check=True,
            )
            if NUM_THREADS := os.environ.get("NUM_THREADS"):
                subprocess.run(["make", f"-j{NUM_THREADS}"], cwd=build_temp, check=True)
            else:
                subprocess.run(["make", "-j"], cwd=build_temp, check=True)
        else:
            raise RuntimeError(f"Unsupported BUILD_WITH={BUILD_WITH}")

    def copy_mlir_bindings(self):
        """Copy built _mlir bindings from build/tools/nkipy/_mlir to nkipy_kernelgen/_mlir"""
        import shutil

        src_dir = os.path.join("build", "tools", "nkipy", "_mlir")
        local_dest = os.path.join("nkipy_kernelgen", "_mlir")

        if os.path.exists(src_dir):
            print(f"Copying _mlir bindings from {src_dir} to {local_dest}")

            # Remove existing destination if it exists
            if os.path.exists(local_dest) or os.path.islink(local_dest):
                if os.path.islink(local_dest) or os.path.isfile(local_dest):
                    os.unlink(local_dest)
                else:
                    shutil.rmtree(local_dest)

            # Copy to local directory
            shutil.copytree(src_dir, local_dest, symlinks=True)
            print(f"Successfully copied _mlir bindings to {local_dest}")
        else:
            print(f"Warning: {src_dir} not found. MLIR bindings may not be available.")


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


if __name__ == "__main__":
    with open("README.md", encoding="utf-8") as fp:
        long_description = fp.read()

    setup(
        name="nkipy-kernelgen",
        description="Lowering from NKIPy to NKI compiler's NISA dialect",
        version="0.1",
        long_description=long_description,
        long_description_content_type="text/markdown",
        setup_requires=["pybind11>=2.8.0", "nanobind>=2.4"],
        install_requires=parse_requirements("requirements.txt"),
        packages=find_packages(),
        ext_modules=[CMakeExtension("mlir", sourcedir="mlir")],
        cmdclass={"build_ext": CMakeBuild},
        python_requires=">=3.10",
        classifiers=[
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering",
            "Topic :: System :: Hardware",
            "Operating System :: OS Independent",
        ],
        zip_safe=False,  # Changed to False to allow including _mlir bindings
        package_data={
            "nkipy_kernelgen._mlir": ["**/*.so", "**/*.py"],
        },
        include_package_data=True,
    )
