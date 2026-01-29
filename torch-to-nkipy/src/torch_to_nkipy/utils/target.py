# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from nkipy.core import compile

CompilationTarget = compile.CompilationTarget

_TRN_1 = "trn1"
_TRN_2 = "trn2"

SUPPORTED_TYPES = {"trn1", "trn1n", "trn2"}


def get_platform_target(compiler_args=None) -> str:
    if "NEURON_PLATFORM_TARGET_OVERRIDE" in os.environ:
        return os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"]

    # If user has supplied the target flag already, use the provided value.
    if compiler_args is not None and "--target" in compiler_args:
        if isinstance(compiler_args, str):
            compiler_args = compiler_args.split()
        index = compiler_args.index("--target")
        target = compiler_args[index + 1]
        assert target in SUPPORTED_TYPES, (
            f"{target} is not a supported platform. \
        Please choose from options trn1, trn1n, or trn2."
        )
        return CompilationTarget.TRN1 if "trn1" in target else CompilationTarget.TRN2

    fpath = "/sys/devices/virtual/dmi/id/product_name"
    try:
        with open(fpath, "r") as f:
            fc = f.readline()
    except IOError:
        raise RuntimeError(
            'Unable to read platform target. If running on CPU, please supply \
        compiler argument target, with one of options trn1, trn1n, or trn2. Ex: \
        "--target trn1"'
        )

    instance_type = fc.split(".")[0]
    if _TRN_1 in instance_type:
        return CompilationTarget.TRN1
    elif _TRN_2 in instance_type:
        return CompilationTarget.TRN2
    else:
        raise RuntimeError(
            f'Unsupported Platform - {fc}. If you want to compile on CPU, \
        please supply compiler argument target, with one of options trn1, \
        trn1n, or trn2. Ex: "--target trn1"'
        )
