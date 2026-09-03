#!/usr/bin/env bash
####### prefill #######
# moe input all gather
nccom-test -r 128 -d bf16 -b 754974720 -e 754974720 --show-input-output-size all_gather 
# moe input reduce scatter
nccom-test -r 128 -d bf16 -b 754974720 -e 754974720 --show-input-output-size reduce_scatter
# all to all
nccom-test -r 128 -d bf16 -b 45M -e 45M --show-input-output-size alltoall --custom-replica-groups scripts/all_to_all_custom_groups.json
####### decode #######
# attn output all reduce

# moe input all gather

# moe output all reduce
nccom-test -r 128 -d bf16 -b 737280 -e 737280 --show-input-output-size all_reduce 


