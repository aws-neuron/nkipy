"""CLI entry point: python -m nkipy.tools.profiler ..."""

import sys

from .merge_profiles import merge_kernel_only, merge_scalene_and_kernel_profiles

USAGE = (
    "Usage: python -m nkipy.tools.profiler "
    "<scalene.json> <kernel_profile.json> <output.json>\n"
    "       python -m nkipy.tools.profiler --kernel-only "
    "<kernel_profile.json> [more...] <output.json>"
)

args = sys.argv[1:]

if "--kernel-only" in args:
    args.remove("--kernel-only")
    if len(args) < 2:
        print(USAGE)
        sys.exit(1)
    output = args[-1]
    kernel_paths = args[:-1]
    merge_kernel_only(kernel_paths, output)
elif len(args) == 3:
    merge_scalene_and_kernel_profiles(args[0], args[1], args[2])
else:
    print(USAGE)
    sys.exit(1)
