#!/usr/bin/env bash
# vast_status.sh — list vast.ai instances and show training progress inside each.
# Usage: ./vast_status.sh              watch mode — refresh every 10s, Ctrl-C to stop
#        ./vast_status.sh -w SECS      watch with a custom interval
#        ./vast_status.sh -1           one-shot (for piping/scripting)
#        LOG_TAIL=20 ./vast_status.sh  longer log tail (default 2)
# Convention: training runs write /workspace/*.log and an exit-code sentinel /workspace/DONE.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VASTAI="/home/kyusang/.conda/envs/dacon/bin/vastai"
PY="/home/kyusang/.conda/envs/dacon/bin/python"
SSH_KEY="$ROOT/.vast/vast_ed25519"
LOG_TAIL="${LOG_TAIL:-2}"

set -a; source "$ROOT/.env"; set +a

show_status() {
json="$("$VASTAI" show instances --raw 2>/dev/null)" || { echo "vastai failed (check VAST_API_KEY in .env)"; return 1; }

echo "=== vast.ai instances ==="
echo "$json" | "$PY" -c '
import json, sys
rows = json.load(sys.stdin)
if not rows:
    print("(no instances)")
    sys.exit(0)
fmt = "{:>9}  {:<10}  {:<12}  {:>7}  {:>7}  {:>7}  {:>4}  {:>4}  {:>5}  {:<22}  {}"
print(fmt.format("ID", "STATUS", "GPU", "$/hr", "AGE", "SPENT", "GPU%", "TEMP", "CPU%", "SSH", "LABEL"))
total = 0.0
def pct(v, f="{:.0f}"):
    return "-" if v is None else f.format(v)
for r in rows:
    ssh = "{}:{}".format(r.get("ssh_host"), r.get("ssh_port"))
    gpu = (r.get("gpu_name") or "?").replace(" ", "_")
    dur = r.get("duration") or 0
    # ~consumed: (GPU+storage rate) x lifetime + billed internet; authoritative = billing page
    spent = dur / 3600 * (r.get("dph_total") or 0) \
            + (r.get("inet_up_cost") or 0) + (r.get("inet_down_cost") or 0)
    total += spent
    age = "{}:{:02d}".format(int(dur // 3600), int(dur % 3600 // 60))
    print(fmt.format(r["id"], r.get("actual_status") or "?", gpu,
                     "{:.3f}".format(r.get("dph_total") or 0), age,
                     "${:.2f}".format(spent),
                     pct(r.get("gpu_util")),
                     "-" if r.get("gpu_temp") is None else "{:.0f}C".format(r["gpu_temp"]),
                     pct(r.get("cpu_util"), "{:.1f}"),
                     ssh, r.get("label") or ""))
    # dashboard-style status message (docker pull progress while loading, then run state)
    for ln in (r.get("status_msg") or "").strip().splitlines():
        print("           status: " + ln.strip()[:150])
if len(rows) > 1:
    print("all instances: ~${:.2f}".format(total))
'

echo "$json" | "$PY" -c '
import json, sys
for r in json.load(sys.stdin):
    if r.get("actual_status") == "running":
        print(r["id"], r.get("ssh_host"), r.get("ssh_port"), r.get("label") or "")
' | while read -r id host port label; do
    echo
    echo "=== ${label:-instance $id} — training progress ==="
    # ControlMaster: first call opens a persistent tunnel (socket in /tmp — NOT NFS,
    # unix sockets break there); later refreshes reuse it => no per-refresh handshake
    # -n: don't let ssh swallow the while-loop's stdin (it would eat the remaining instances)
    ssh -qn -i "$SSH_KEY" -p "$port" "root@$host" \
        -o ControlMaster=auto -o ControlPath="/tmp/vast-cm-%r@%h-%p" -o ControlPersist=120 \
        -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes '
        if [ -f /workspace/DONE ]; then
            code=$(cat /workspace/DONE)
            if [ "$code" = "0" ]; then echo "STATUS: FINISHED (exit 0)"; else echo "STATUS: FAILED (exit $code)"; fi
        elif pgrep -f "src\.finetune|mlm_tapt|python.*train" >/dev/null 2>&1; then
            echo "STATUS: RUNNING"
        else
            echo "STATUS: idle — no training process, no DONE sentinel"
        fi
        echo "--- gpu ---"
        nvidia-smi --query-gpu=index,pci.bus_id,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null \
            | awk -F", " "
                BEGIN {printf \"%-3s  %-9s  %5s  %16s  %13s  %5s\n\", \"GPU\", \"BUS\", \"UTIL\", \"MEM-USAGE\", \"PWR:USE/CAP\", \"TEMP\"}
                {sub(/^0+:/, \"\", \$2)
                 printf \"%-3s  %-9s  %4s%%  %10s/%-5s  %6.0f/%-4.0fW  %4sC\n\", \$1, \$2, \$3, \$4, \$5, \$6, \$7, \$8}" \
            || echo "(nvidia-smi unavailable)"
        gp=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
            | awk -F", " "{sub(/^0+:/, \"\", \$1); printf \"  bus %-9s  pid %-7s  %-8s  %s\n\", \$1, \$2, \$3, \$4}")
        echo "procs on gpu:"
        echo "${gp:-  (none)}"
        echo "--- python (workers/wrappers filtered) ---"
        pl=$(ps -eo pid,pcpu,etime,args --sort=-pcpu | grep "[p]ython" \
            | grep -vE "compile_worker|_inductor|triton|pgrep|bash -c|sh -c" \
            | head -6 | cut -c1-150)
        echo "${pl:-(none)}"
        log=$(ls -t /workspace/*.log 2>/dev/null | head -1)
        if [ -n "$log" ]; then
            echo "--- tail -n '"$LOG_TAIL"' $log ---"
            # tr: tqdm progress bars use \r; in a redirected log they pile into one
            # giant line — split them so tail shows the latest bar state as lines
            tail -c 65536 "$log" | tr "\r" "\n" | tail -n '"$LOG_TAIL"'
        else
            echo "(no /workspace/*.log yet)"
        fi
    ' || echo "(ssh failed — instance may still be booting or key not attached)"
done
}

if [ "${1:-}" = "-1" ]; then
    show_status
else
    INTERVAL=10
    [ "${1:-}" = "-w" ] && INTERVAL="${2:-10}"
    while true; do
        out="$(show_status 2>&1)"   # render off-screen, swap in one go (no flicker)
        clear
        echo "vast_status — $(date '+%H:%M:%S') — refresh ${INTERVAL}s, Ctrl-C to stop"
        printf '%s\n' "$out"
        sleep "$INTERVAL"
    done
fi
