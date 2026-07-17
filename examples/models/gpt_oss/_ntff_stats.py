"""Extract device exec-time + op counts from a .ntff, auto-matching its NEFF."""
import glob, json, subprocess, sys, os

def find_neff_and_view(ntff, patterns):
    for pat in patterns:
        for neff in sorted(glob.glob(pat), key=os.path.getmtime, reverse=True):
            r = subprocess.run(
                ["neuron-profile", "view", "-n", neff, "-s", ntff,
                 "--output-format", "json", "--output-file", "/tmp/_ntff.json"],
                capture_output=True, text=True)
            if os.path.exists("/tmp/_ntff.json") and os.path.getsize("/tmp/_ntff.json") > 1000 \
               and "Unable to process" not in (r.stderr + r.stdout):
                return neff, json.load(open("/tmp/_ntff.json"))
            if os.path.exists("/tmp/_ntff.json"):
                os.remove("/tmp/_ntff.json")
    return None, None

def stats(name, ntff, patterns):
    neff, d = find_neff_and_view(ntff, patterns)
    if d is None:
        print(f"{name}: NO NEFF MATCH"); return
    ins = d["instruction"]
    busy = [i for i in ins if i.get("duration", 0)]
    t0 = min(i["timestamp"] for i in busy)
    t1 = max(i["timestamp"] + i["duration"] for i in busy)
    mm = [i for i in ins if i.get("opcode") == "MATMUL"]
    dma = len(d.get("dma", []))
    print(f"{name}: neff={os.path.basename(os.path.dirname(neff))}")
    print(f"   device wall = {(t1-t0)/1000:8.1f} us | matmuls={len(mm):5d} | "
          f"instrs={len(ins):5d} | dma_recs={dma}")
    return (t1 - t0) / 1000

if __name__ == "__main__":
    b = stats("baseline", "moe_base.ntff", ["build/moe_baseline_*/*.neff"])
    t = stats("batched ", "moe_batch.ntff", ["build/moe_batched_*/*.neff"])
    if b and t:
        print(f"\nDEVICE speedup: {b/t:.2f}x  ({b:.1f}us -> {t:.1f}us)")
