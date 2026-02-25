#!/usr/bin/env bash

function profile-upload() {
    prof_server_url="https://neuron-profiler.corp.amazon.com"
    common_flags=( -L --cookie ~/.midway/cookie --cookie-jar ~/.midway/cookie )

    # authenticate with midway
    if ! curl "${common_flags[@]}" --fail-with-body "${prof_server_url}" > /dev/null 2>&1; then
        echo "midway auth failed. You need to run mwinit"
    else
        # upload profile
        curl "${common_flags[@]}" "${prof_server_url}/api/upload" "$@"
    fi
}

# Default tag value
DEFAULT_TAG="zhenyus"
PROFILE_TAG="$DEFAULT_TAG"
DIRECTORIES=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            PROFILE_TAG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-t|--tag TAG] <dir1> [dir2] [dir3] ..."
            echo "  -t, --tag TAG    Profile tag (default: $DEFAULT_TAG)"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Example: $0 build/fused_attention_router build/blockwise_nki_static"
            echo "Example: $0 -t custom_tag_name build/fused_attention_router"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
        *)
            DIRECTORIES+=("$1")
            shift
            ;;
    esac
done

# Check if at least one directory argument is provided
if [ ${#DIRECTORIES[@]} -eq 0 ]; then
    echo "Usage: $0 [-t|--tag TAG] <dir1> [dir2] [dir3] ..."
    echo "  -t, --tag TAG    Profile tag (default: $DEFAULT_TAG)"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Example: $0 build/fused_attention_router build/blockwise_nki_static"
    echo "Example: $0 -t custom_tag_name build/fused_attention_router"
    exit 0
fi

# Loop through all provided directory arguments
for dir in "${DIRECTORIES[@]}"; do
    if [[ -d "$dir" ]]; then
        dir_basename=$(basename "$dir")
        NEFF_FILE="${dir}/${dir_basename}.neff"
        NTFF_FILE="${dir}/profile.ntff"
        
        # Check if both required files exist
        if [[ -f "$NEFF_FILE" && -f "$NTFF_FILE" ]]; then
            # PROFILE_NAME="zhenyus_qwen3_debug_lnc2_not_syncv_${dir_basename}"
            PROFILE_NAME="${PROFILE_TAG}_${dir_basename}"
            echo "Uploading profile for ${dir} with name: ${PROFILE_NAME}"
            profile-upload -F neff=@${NEFF_FILE} -F ntff=@${NTFF_FILE} -F name=${PROFILE_NAME} -F display-name=${dir_basename} -F "force=true"
        else
            echo "Skipping ${dir}: missing required files (file.neff and/or profile.ntff)"
        fi
    else
        echo "Skipping ${dir}: directory does not exist"
    fi
done