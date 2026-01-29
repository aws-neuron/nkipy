# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import List


@dataclass
class NtffMeta:
    save_ntff: bool = False
    save_ntff_dir: str = "./ntff_dir"
    save_ntff_exe_idx: List[int] = field(default_factory=list)

    kernel_hash: str = ""

    curr_exe_idx: int = 0

    @classmethod
    def from_options_and_kernel_hash(
        cls, options: dict | None, kernel_hash: str
    ) -> "NtffMeta":
        if options is None:
            options = {}
        return cls(
            save_ntff=options.get("save_ntff", False),
            save_ntff_dir=options.get("save_ntff_dir", "./ntff_dir"),
            save_ntff_exe_idx=options.get("save_ntff_exe_idx", []),
            kernel_hash=kernel_hash,
        )
