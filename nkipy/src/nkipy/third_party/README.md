# Third-Party Libraries

This directory contains third-party code used by NKIPy.

## XLA (OpenXLA)

The `xla/` directory contains protobuf definitions from the [OpenXLA project](https://github.com/openxla/xla).

**Source:** https://github.com/openxla/xla  
**License:** Apache License 2.0  
**Files:**
- `xla/xla_data.proto` - XLA data types and shape definitions
- `xla/service/hlo.proto` - HLO instruction definitions
- `xla/service/metrics.proto` - Metrics definitions

These files are used to generate protobuf bindings for serializing HLO graphs.
