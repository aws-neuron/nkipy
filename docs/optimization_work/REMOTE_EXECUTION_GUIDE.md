# Remote Execution Guide for Instance B

## The Correct Way to Run Commands on Instance B

### Instance Information
- **Instance A (Server)**: 172.31.44.131 - `/home/ubuntu/vllm-nkipy/nkipy`
- **Instance B (Receiver)**: 172.31.40.200 - `/home/ubuntu/vllm-nkipy/nkipy`

### Critical Requirements ⚠️

When executing commands remotely via SSH, you **MUST**:

1. **Change to the correct directory FIRST**
2. **Activate the virtual environment** (if needed for Python commands)
3. **Use absolute paths** or ensure the working directory is set

## Common Mistakes to Avoid ❌

### ❌ WRONG - No directory change:
```bash
ssh ubuntu@172.31.40.200 "source .venv/bin/activate && bash script.sh"
# ERROR: .venv/bin/activate: No such file or directory
```

### ❌ WRONG - Relative path without cd:
```bash
ssh ubuntu@172.31.40.200 "bash run_remote_hypothesis_test.sh"
# ERROR: run_remote_hypothesis_test.sh: No such file or directory
```

### ❌ WRONG - Activating venv before cd:
```bash
ssh ubuntu@172.31.40.200 "source ~/vllm-nkipy/nkipy/.venv/bin/activate && bash script.sh"
# ERROR: script.sh: No such file or directory (wrong directory)
```

## Correct Patterns ✅

### ✅ Pattern 1: cd THEN activate THEN run
```bash
ssh ubuntu@172.31.40.200 "cd /home/ubuntu/vllm-nkipy/nkipy && source .venv/bin/activate && bash run_remote_hypothesis_test.sh"
```

### ✅ Pattern 2: Use script that handles cd internally
```bash
ssh ubuntu@172.31.40.200 "cd /home/ubuntu/vllm-nkipy/nkipy && bash run_remote_hypothesis_test.sh"
# Where run_remote_hypothesis_test.sh already has: cd /home/ubuntu/vllm-nkipy/nkipy
```

### ✅ Pattern 3: Interactive SSH (most reliable for testing)
```bash
ssh ubuntu@172.31.40.200
# Then on Instance B:
cd /home/ubuntu/vllm-nkipy/nkipy
source .venv/bin/activate
bash run_remote_hypothesis_test.sh
```

## Standard Commands for Instance B

### Start Receiver Engine
```bash
# From Instance A
ssh ubuntu@172.31.40.200 "cd /home/ubuntu/vllm-nkipy/nkipy && source .venv/bin/activate && bash examples/p2p/run_vllm_qwen_1_receiver.sh > receiver.log 2>&1 &"

# Or interactively on Instance B
cd /home/ubuntu/vllm-nkipy/nkipy
source .venv/bin/activate
bash examples/p2p/run_vllm_qwen_1_receiver.sh > receiver.log 2>&1 &
```

### Check Receiver Status
```bash
# From Instance A
ssh ubuntu@172.31.40.200 "curl -s http://localhost:8000/health"

# Check logs
ssh ubuntu@172.31.40.200 "tail -50 /home/ubuntu/vllm-nkipy/nkipy/receiver.log"
```

### Kill Receiver Engine
```bash
# From Instance A
ssh ubuntu@172.31.40.200 "fuser -k 8000/tcp; pkill -9 -f vllm_plugin.server"
```

### Sync Code to Instance B
```bash
# From Instance A (current directory: /home/ubuntu/vllm-nkipy/nkipy)
rsync -az nkipy/src/nkipy/ ubuntu@172.31.40.200:/home/ubuntu/vllm-nkipy/nkipy/nkipy/src/nkipy/
rsync -az examples/p2p/ ubuntu@172.31.40.200:/home/ubuntu/vllm-nkipy/nkipy/examples/p2p/

# Or sync everything (use carefully)
rsync -az --exclude '.venv' --exclude '*.log' . ubuntu@172.31.40.200:/home/ubuntu/vllm-nkipy/nkipy/
```

## Hypothesis Test Execution

### Complete Command (from Instance A)
```bash
ssh ubuntu@172.31.40.200 "cd /home/ubuntu/vllm-nkipy/nkipy && source .venv/bin/activate && bash run_remote_hypothesis_test.sh"
```

### Step-by-Step (Interactive - RECOMMENDED for testing)
```bash
# Step 1: SSH to Instance B
ssh ubuntu@172.31.40.200

# Step 2: Navigate to repo
cd /home/ubuntu/vllm-nkipy/nkipy

# Step 3: Activate venv
source .venv/bin/activate

# Step 4: Run test
bash run_remote_hypothesis_test.sh
```

## Key Lessons Learned

1. **SSH remote commands start in the HOME directory** (`/home/ubuntu`), NOT the repo directory
2. **Relative paths don't work** unless you cd first
3. **Virtual environment activation** requires being in the correct directory first
4. **Always use the pattern**: `cd DIR && source VENV && RUN COMMAND`
5. **For complex operations**: Use interactive SSH instead of one-liner commands

## Testing Checklist

Before running tests on Instance B:

- [ ] Server running on Instance A (172.31.44.131:8000)
- [ ] Changed to correct directory: `cd /home/ubuntu/vllm-nkipy/nkipy`
- [ ] Activated venv: `source .venv/bin/activate`
- [ ] Scripts are executable: `chmod +x *.sh`
- [ ] Scripts are synced from Instance A if modified

## Emergency Recovery

If Instance B gets stuck:
```bash
# From Instance A
ssh ubuntu@172.31.40.200 "fuser -k 8000/tcp; pkill -9 -f vllm_plugin.server; pkill -9 -f nkipy"
# Wait 5 seconds
ssh ubuntu@172.31.40.200 "cd /home/ubuntu/vllm-nkipy/nkipy && source .venv/bin/activate && bash examples/p2p/run_vllm_qwen_1_receiver.sh > receiver.log 2>&1 &"
```
