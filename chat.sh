# export NEURON_RT_LOG_LEVEL_NRT=DEBUG

# disable warning
export OMP_NUM_THREADS=1
export TP_SIZE=8

export DP_SIZE=16
export PREFILL_EP_SIZE=4

######################## minimal config for fast dev ########################
# export DP_SIZE=2
# export PREFILL_EP_SIZE=2
# export NEURON_LOGICAL_NC_CONFIG=2
# export NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=52428800 # increase buffer size for large collective operations
#############################################################################

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rm -rf ./profile
# numactl --cpunodebind=0 --membind=0 
lsof -ti:29501 | xargs -r kill -9 # kill previous run
torchrun --nproc-per-node=$((TP_SIZE*DP_SIZE)) --master-port=29501 $SCRIPT_DIR/chat.py --tp_size=$TP_SIZE --prefill_ep_size=$PREFILL_EP_SIZE 2>&1 | tee chat.log
neuron-profile view -d profile/ --output-format perfetto # -v 4
# ./upload_profile.sh -t tmp /tmp/build/*