#!/usr/bin/env bash
# Read-only context snapshot for a new thesis-project agent/session.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

section() {
  printf '\n== %s ==\n' "$1"
}

section "Repository"
printf 'root: %s\n' "$ROOT"
git status -sb
git log -1 --format='commit: %h %s (%ci)'
if [ -f .gitmodules ]; then
  git submodule status || true
fi

section "Required onboarding"
for path in AGENTS.md docs/START_HERE.md experiments/registry.yaml results/current/manifest.json; do
  if [ -e "$path" ]; then
    printf 'OK      %s\n' "$path"
  else
    printf 'MISSING %s\n' "$path"
  fi
done

section "Experiment registry"
registry_rows=$(awk '
  function clean(value) {
    gsub(/^"|"$/, "", value)
    return value
  }
  function emit() {
    if (id != "") {
      printf "%s|%s|%s|%s|%s\n", id, status, stage, log_path, metrics
    }
  }
  /^experiments:/ {inside=1; next}
  /^paper_targets:/ {emit(); id=""; exit}
  inside && /^  [a-zA-Z0-9_-]+:$/ {
    emit()
    id=$1
    sub(/:$/, "", id)
    status="unknown"
    stage="-"
    log_path=""
    metrics=""
    next
  }
  inside && /^    status:/ {status=clean($2); next}
  inside && /^    stage:/ {stage=clean($2); next}
  inside && /^    log:/ {
    log_path=$0
    sub(/^    log:[[:space:]]*/, "", log_path)
    log_path=clean(log_path)
    next
  }
  inside && /^    metrics:/ {
    metrics=$0
    sub(/^    metrics:[[:space:]]*/, "", metrics)
    metrics=clean(metrics)
    next
  }
  END {if (inside) emit()}
' experiments/registry.yaml)

while IFS='|' read -r exp_id status stage _ metrics_path; do
  [ -n "$exp_id" ] || continue
  printf '%-34s status=%-8s stage=%-16s metrics=%s\n' \
    "$exp_id" "$status" "$stage" "$metrics_path"
done <<< "$registry_rows"

while IFS='|' read -r exp_id status _ log_path metrics_path; do
  [ "$status" = "running" ] || continue
  section "Live run: $exp_id"
  if [ -n "$log_path" ] && [ -f "$log_path" ]; then
    tail -8 "$log_path"
    log_dir=$(dirname "$log_path")
    for shard_log in "$log_dir"/knowledge_shard*.log; do
      [ -f "$shard_log" ] || continue
      progress=$(tr '\r' '\n' < "$shard_log" | grep -E 'llm-(items|users)' | tail -1 || true)
      progress=${progress%%The following generation flags*}
      [ -n "$progress" ] && printf '%s: %s\n' "$(basename "$shard_log")" "$progress"
    done
  else
    printf 'log unavailable: %s\n' "${log_path:-not registered}"
  fi
  if [ -n "$metrics_path" ] && [ -f "$metrics_path" ]; then
    printf 'metrics already exist: %s\n' "$metrics_path"
  else
    printf 'final metrics not present yet: %s\n' "${metrics_path:-not registered}"
  fi
done <<< "$registry_rows"

section "GPU visibility"
if gpu_status=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null); then
  printf '%s\n' "$gpu_status"
else
  echo "GPU status unavailable in current namespace; use logs/registry as evidence."
fi

section "Read next"
cat <<'EOF'
Always: docs/START_HERE.md
UR4Rec: docs/UR4Rec_code_and_reproduction.md
Exp3RT: docs/exp3rt_reproduction.md
Avito/C-UR4Rec: docs/avito_preferences.md
Literature/task 26.06: docs/task_2026-06-26_artem.md
Historical chronology only: docs/AGENT_HANDOFF.md
EOF
