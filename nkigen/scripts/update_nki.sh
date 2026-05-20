#!/bin/bash
# Update NKI: pull latest, reinstall neuronx-cc, rebuild, run tests.
#
# Usage:
#   source update_nki.sh             # pull + neuronx-cc + rebuild + check-nki
#   source update_nki.sh --skip-cc   # skip neuronx-cc reinstall
#   source update_nki.sh --skip-test # skip check-nki tests
set -eo pipefail

NKI_SRC="/home/ubuntu/nki-src"
VENV_PATH="/opt/nki-venv"
PYTHON="$VENV_PATH/bin/python3"
AWS_REGION="us-west-2"
DOMAIN_OWNER="149122183214"
ROLE_ARN="arn:aws:iam::039612851861:role/Kdev-Artifacts-Consumer-Development"

RUN_TESTS=true
SKIP_CC=false
for arg in "$@"; do
    case "$arg" in
        --skip-test) RUN_TESTS=false ;;
        --skip-cc)   SKIP_CC=true ;;
        *) echo "Unknown arg: $arg"; return 1 2>/dev/null || exit 1 ;;
    esac
done

# --- 1. Pull latest ---
echo "==> Pulling latest NKI source..."
git -C "$NKI_SRC" pull --ff-only

# --- 2. Reinstall neuronx-cc ---
if [ "$SKIP_CC" = false ]; then
    NEURONX_CC_VERSION=$(cat "$NKI_SRC/.ci/baremetal/NEURONX_CC_VERSION")
    echo "==> Assuming IAM role for CodeArtifact..."
    CREDS=$(aws sts assume-role \
        --role-arn "$ROLE_ARN" \
        --role-session-name nki-update \
        --region "$AWS_REGION" \
        --output json)
    export AWS_ACCESS_KEY_ID=$(echo "$CREDS" | python3 -c "import sys,json; print(json.load(sys.stdin)['Credentials']['AccessKeyId'])")
    export AWS_SECRET_ACCESS_KEY=$(echo "$CREDS" | python3 -c "import sys,json; print(json.load(sys.stdin)['Credentials']['SecretAccessKey'])")
    export AWS_SESSION_TOKEN=$(echo "$CREDS" | python3 -c "import sys,json; print(json.load(sys.stdin)['Credentials']['SessionToken'])")

    echo "==> Logging into CodeArtifact..."
    aws codeartifact login --tool pip \
        --repository kdev-artifacts-development \
        --domain amazon \
        --domain-owner "$DOMAIN_OWNER" \
        --region "$AWS_REGION"

    echo "==> Installing neuronx-cc==${NEURONX_CC_VERSION}..."
    "$PYTHON" -m pip install --no-cache-dir "neuronx-cc==${NEURONX_CC_VERSION}"

    # Scrub CodeArtifact pip config
    rm -f ~/.config/pip/pip.conf ~/.pip/pip.conf /etc/pip.conf 2>/dev/null || true
else
    echo "==> Skipping neuronx-cc install (--skip-cc)"
fi

# --- 3. Rebuild NKI ---
echo "==> Rebuilding NKI..."
source "$NKI_SRC/scripts/dev_setup.sh"
nki-rebuild

# After nki-rebuild, Python binding is available via PYTHONPATH which
# dev_setup.sh sets to include build/python/py-staging. Verify:
echo "==> Verifying NKI Python binding..."
"$PYTHON" -c "import nki; print(f'nki imported from: {nki.__file__}')"

# --- 4. Tests ---
if [ "$RUN_TESTS" = true ]; then
    echo "==> Running check-nki..."
    ninja -C "$NKI_SRC/build" check-nki
else
    echo "==> Skipping tests (--skip-test)"
fi

echo "==> Done."
