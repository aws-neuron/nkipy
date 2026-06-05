#!/usr/bin/env python3
"""Benchmark: 3-instance broadcast (Qwen3-235B, separate instances, no contention)."""
import concurrent.futures, os, signal, subprocess, sys, tempfile, time, requests
sys.stdout.reconfigure(line_buffering=True)
os.environ["VLLM_PLUGINS"] = "nkipy"

LOCAL_IP = "10.3.211.148"
RECV_HOSTS = ["10.3.215.3", "10.3.220.183"]
MODEL = "Qwen/Qwen3-235B-A22B"
CHECKPOINT = "/fsx/zhuangw/models/qwen3_235b_a22b_TP32"
TP = 32
PY = "/fsx/zhuangw/nkipy/.venv/bin/python"
SP = 8100
RP = 9100

def wait(h, p, t=900):
    d = time.time() + t
    while time.time() < d:
        try:
            if requests.get(f"http://{h}:{p}/nkipy/health", timeout=2).status_code == 200:
                return True
        except:
            pass
        time.sleep(5)
    return False

def start_s():
    e = os.environ.copy()
    e.update({"VLLM_PLUGINS": "nkipy", "VLLM_USE_V1": "1", "NKIPY_CORE_OFFSET": "0",
              "NKIPY_CHECKPOINT": CHECKPOINT, "OMP_NUM_THREADS": "1",
              "VLLM_RPC_TIMEOUT": "600000", "HF_HUB_OFFLINE": "1", "NEURON_RT_MAP_HBM": "1",
              "PATH": f"{os.path.dirname(PY)}:{e.get('PATH', '')}"})
    l = tempfile.NamedTemporaryFile(mode="w", prefix="b3s_", suffix=".log", delete=False)
    return subprocess.Popen([PY, "-m", "nkipy.vllm_plugin.server", "--model", MODEL,
        "--tensor-parallel-size", str(TP), "--max-model-len", "128", "--max-num-seqs", "1",
        "--enforce-eager", "--dtype", "bfloat16", "--port", str(SP)],
        env=e, stdout=l, stderr=subprocess.STDOUT)

def start_r(h):
    vb = os.path.dirname(PY)
    ev = (f"PATH={vb}:$PATH VLLM_PLUGINS=nkipy VLLM_USE_V1=1 NKIPY_CORE_OFFSET=0 "
          f"OMP_NUM_THREADS=1 VLLM_RPC_TIMEOUT=600000 HF_HUB_OFFLINE=1 NEURON_RT_MAP_HBM=1")
    cm = (f"{PY} -m nkipy.vllm_plugin.server --model {MODEL} --tensor-parallel-size {TP} "
          f"--max-model-len 128 --max-num-seqs 1 --enforce-eager --dtype bfloat16 --port {RP}")
    l = tempfile.NamedTemporaryFile(mode="w", prefix=f"b3r{h[-3:]}_", suffix=".log", delete=False)
    return subprocess.Popen(["ssh", "-o", "StrictHostKeyChecking=no", h, f"{ev} {cm}"],
        stdout=l, stderr=subprocess.STDOUT)

def wake(h):
    t0 = time.time()
    r = requests.post(f"http://{h}:{RP}/nkipy/wake_up", json={"peer_url": None}, timeout=600)
    if r.status_code != 200:
        print(f"FAIL {h}: {r.text[:300]}")
        return None
    res = r.json()
    res["_t"] = time.time() - t0
    return res

def infer(h, p):
    r = requests.post(f"http://{h}:{p}/v1/completions",
        json={"model": MODEL, "prompt": "The capital of France is",
              "max_tokens": 20, "temperature": 0}, timeout=180)
    if r.status_code != 200:
        return f"ERROR:{r.status_code}:{r.text[:100]}"
    choices = r.json().get("choices", [])
    return choices[0].get("text", "NO_TEXT") if choices else "NO_CHOICES"

ps = []
try:
    print("Starting engines...")
    ps.append(start_s())
    for h in RECV_HOSTS:
        ps.append(start_r(h))

    print("Waiting sender...")
    assert wait("localhost", SP), "Sender failed"
    print("Sender OK")
    for h in RECV_HOSTS:
        print(f"Waiting {h}...")
        assert wait(h, RP), f"{h} failed"
        print(f"  OK")

    ref = infer("localhost", SP)
    print(f"Ref: {ref[:50]!r}\n")

    for iteration in range(3):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration+1}/3")
        print(f"{'='*60}")

        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            rs = list(pool.map(wake, RECV_HOSTS))
        tw = time.time() - t0
        ms = []
        for h, r in zip(RECV_HOSTS, rs):
            if r is None:
                print("  ABORT")
                sys.exit(1)
            lat = r.get("latency", {})
            print(f"  Wake {h}: {r['_t']:.2f}s (nrt={lat.get('nrt_init_s','?')}s, total={lat.get('total_s','?')}s)")
            ms.append(r.get("rdma_metadata"))
        print(f"  Wake wall: {tw:.2f}s")

        t0 = time.time()
        rp = requests.post(f"http://localhost:{SP}/nkipy/push_weights",
                           json={"receivers": ms}, timeout=300)
        tx = time.time() - t0
        print(f"  Transfer: {tx:.2f}s ({rp.json().get('status')})")

        ok = True
        for h in RECV_HOSTS:
            o = infer(h, RP)
            p = (o == ref) if not o.startswith("ERROR") else False
            if not p:
                ok = False
            print(f"  Infer {h}: {'PASS' if p else 'FAIL'} — {o[:40]!r}")

        print(f"  TOTAL: wake={tw:.2f}s + transfer={tx:.2f}s = {tw+tx:.2f}s | {'PASS' if ok else 'FAIL'}")

        # Sleep both
        for h in RECV_HOSTS:
            requests.post(f"http://{h}:{RP}/nkipy/sleep", timeout=60)
        time.sleep(3)

finally:
    for p in ps:
        p.send_signal(signal.SIGTERM)
    for p in ps:
        try:
            p.wait(timeout=30)
        except:
            p.kill()
    for h in RECV_HOSTS:
        subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", h,
            "pkill -f 'nkipy.vllm_plugin.server' 2>/dev/null || true"], capture_output=True)
