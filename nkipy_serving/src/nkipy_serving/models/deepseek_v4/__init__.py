"""DeepSeek-V4-Flash model package."""

from nkipy_serving.models.deepseek_v4.assembly import (
    topology as assembly_topology,
)
from nkipy_serving.models.deepseek_v4.config import (
    DeepseekV4ModelConfig,
    DeepseekV4Weights,
)
from nkipy_serving.models.deepseek_v4.executor import DeepseekV4Executor
from nkipy_serving.models.deepseek_v4.rank_layout import (
    V4LaneRoute,
    V4RankCoord,
    build_attention_dp_lane_routes,
    build_moe_ep_row_groups,
    build_replica_groups,
    build_tp_row_groups,
    coord_for_rank,
    local_expert_ids,
    validate_v4_rank_layout,
)
from nkipy_serving.models.deepseek_v4.weights import (
    get_deepseek_v4_kv_metadata,
    init_deepseek_v4_weights,
)

__all__ = [
    "DeepseekV4ModelConfig",
    "DeepseekV4Weights",
    "DeepseekV4Executor",
    "assembly_topology",
    "get_deepseek_v4_kv_metadata",
    "init_deepseek_v4_weights",
    "V4LaneRoute",
    "V4RankCoord",
    "build_attention_dp_lane_routes",
    "build_moe_ep_row_groups",
    "build_replica_groups",
    "build_tp_row_groups",
    "coord_for_rank",
    "local_expert_ids",
    "validate_v4_rank_layout",
]
