# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_packages, setup

setup(
    name="torch_to_nkipy",
    packages=find_packages(
        where="src",
        include=["torch_to_nkipy", "torch_to_nkipy.*"],
    ),
    package_dir={"": "src"},
    zip_safe=True,
)
