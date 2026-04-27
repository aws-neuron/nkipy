# Re-export from system MLIR to avoid conflicts
from mlir.dialects._ods_common import *
from mlir.dialects._ods_common import (
    _cext,
    segmented_accessor,
    equally_sized_accessor,
    get_default_loc_context,
    get_op_result_or_value,
    get_op_results_or_values,
    get_op_result_or_op_results,
)
