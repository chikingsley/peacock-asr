# Peacock-ASR RunPod + Training Commands
# Usage: just <recipe> [args]

# --- Config ---
template_id := "a1hkwx3tzh"
default_gpu := "NVIDIA RTX A4000"
default_volume := "200"
ssh_key := "~/.ssh/id_ed25519"
ssh_opts := "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"

# Show all recipes
default:
    @just --list

# --- Pod lifecycle ---

# Create a new training pod (e.g. just pod-create "NVIDIA L4" 500)
pod-create gpu=default_gpu volume=default_volume:
    runpodctl pod create \
        --template-id {{template_id}} \
        --gpu-id "{{gpu}}" \
        --gpu-count 1 \
        --volume-in-gb {{volume}}

# List all pods with key info
pod-list:
    @runpodctl pod list 2>&1 | python3 -c "import json,sys; pods=json.load(sys.stdin); [print(f'{p[\"id\"]}  {p[\"name\"]:30s}  \${p[\"costPerHr\"]}/hr  gpu:{p[\"gpuCount\"]}  vol:{p[\"volumeInGb\"]}GB  {p[\"desiredStatus\"]}') for p in pods]"

# Get SSH command for a pod
pod-ssh id:
    @just _ssh-cmd {{id}}

# Open interactive SSH to a pod
pod-shell id:
    #!/usr/bin/env bash
    eval "$(just _ssh-cmd {{id}})"

# Run a command on a pod
pod-exec id +cmd:
    @just _ssh {{id}} "{{cmd}}"

# Quick status: GPU util, VRAM, last 5 log lines
pod-status id:
    @just _ssh {{id}} "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null; echo '---'; { cat /root/train-full.log 2>/dev/null || cat /root/train.log 2>/dev/null; } | tail -c 2000 | tr '\r' '\n' | tail -5"

# Full status: GPU, disk, env, processes, log
pod-info id:
    @just _ssh {{id}} "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null; echo '===DISK==='; df -h /workspace; echo '===ENV==='; cat /workspace/peacock-asr/.env 2>/dev/null | grep -E 'WANDB|HF_TOKEN' | sed 's/=.*/=<set>/'; echo '===PROCS==='; ps aux | grep train_phoneme | grep -v grep || echo 'no training running'; echo '===LOG==='; { cat /root/train-full.log 2>/dev/null || cat /root/train.log 2>/dev/null; } | tail -c 3000 | tr '\r' '\n' | tail -15"

# Status of all running pods at once
status-all:
    #!/usr/bin/env bash
    for id in $(runpodctl pod list 2>&1 | python3 -c "import json,sys; [print(p['id']) for p in json.load(sys.stdin) if p['gpuCount']>0]"); do
        echo "=== $id ==="
        just pod-status "$id" 2>/dev/null || echo "  (unreachable)"
        echo
    done

# Stop a pod
pod-stop id:
    runpodctl pod stop {{id}}

# Start a stopped pod
pod-start id:
    runpodctl pod start {{id}}

# Destroy a pod (stop + remove)
pod-destroy id:
    runpodctl pod stop {{id}} && runpodctl pod remove {{id}}

# --- Training ---

# Launch training on a pod (pass any args to train_phoneme_head.py)
train id +args:
    @just _ssh {{id}} "cd /workspace/peacock-asr && nohup bash -c 'set -a; source .env; set +a; export PATH=/root/.local/bin:\$PATH; export HF_HOME=/workspace/hf_cache; .venv/bin/python -u training/train_phoneme_head.py {{args}}' > /root/train.log 2>&1 & echo PID:\$!"

# Launch wav2vec2-base training (Track 10)
train-wav2vec2 id:
    just train {{id}} --model-name facebook/wav2vec2-base --output-dir /workspace/wav2vec2-base-phoneme-en --hub-repo Peacockery/wav2vec2-base-phoneme-en --batch-size 8 --gradient-accumulation 8 --learning-rate 3e-5 --num-epochs 3 --dataloader-workers 4

# Launch w2v-bert-2.0 training with preprocessed data
train-w2vbert id:
    just train {{id}} --model-name facebook/w2v-bert-2.0 --preprocessed-dataset Peacockery/librispeech-phoneme-features --output-dir /workspace/w2v-bert-phoneme-en --hub-repo Peacockery/w2v-bert-phoneme-en --batch-size 4 --gradient-accumulation 16 --dataloader-workers 8

# Kill training on a pod
train-stop id:
    @just _ssh {{id}} "pkill -f train_phoneme_head && echo 'stopped' || echo 'nothing running'"

# Tail training log (streams live)
train-log id:
    #!/usr/bin/env bash
    CMD=$(just _ssh-cmd {{id}})
    LOG=$(just pod-exec {{id}} "ls -t /root/train-full.log /root/train.log 2>/dev/null | head -1" 2>&1 | grep -v Warning | tr -d '[:space:]')
    [ -z "$LOG" ] && LOG="/root/train.log"
    eval "$CMD -t 'tail -f $LOG | stdbuf -oL tr \"\\r\" \"\\n\"'"

# --- Sync ---

# Rsync local repo to pod
pod-sync id:
    #!/usr/bin/env bash
    set -e
    INFO=$(runpodctl ssh info {{id}} 2>&1)
    HOST=$(echo "$INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['ip'])")
    PORT=$(echo "$INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['port'])")
    rsync -avz \
        --exclude='.venv' --exclude='wandb' --exclude='runs/output' \
        --exclude='processed-features' --exclude='references' --exclude='.git' \
        --exclude='.agents' --exclude='.claude' --exclude='.pi' \
        -e "ssh {{ssh_opts}} -p $PORT -i {{ssh_key}}" \
        . root@$HOST:/workspace/peacock-asr/

# --- GPUs ---

# List available GPUs (default: <= 24GB)
gpus max_gb="24":
    @runpodctl gpu list 2>&1 | python3 -c "import json,sys; [print(f'{g[\"displayName\"]:20s} {g[\"memoryInGb\"]}GB  stock:{g.get(\"stockStatus\",\"?\"):8s}  id: {g[\"gpuId\"]}') for g in json.load(sys.stdin) if g.get('available') and g.get('memoryInGb',0) <= {{max_gb}}]"

# --- Internal helpers ---

# Get SSH command string for a pod
_ssh-cmd id:
    @runpodctl ssh info {{id}} 2>&1 | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'ssh {{ssh_opts}} root@{d[\"ip\"]} -p {d[\"port\"]} -i {{ssh_key}}')" 2>/dev/null || echo "echo 'Pod not ready'"

_ssh id +cmd:
    #!/usr/bin/env bash
    set -e
    INFO=$(runpodctl ssh info {{id}} 2>&1)
    HOST=$(echo "$INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['ip'])")
    PORT=$(echo "$INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['port'])")
    ssh {{ssh_opts}} root@$HOST -p $PORT -i {{ssh_key}} "{{cmd}}"
