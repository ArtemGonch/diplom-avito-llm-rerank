#!/usr/bin/env bash
# Create public GitHub repo and push (requires GITHUB_TOKEN with repo scope).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REPO_NAME="${GITHUB_REPO_NAME:-diplom-avito-llm-rerank}"
GITHUB_USER="${GITHUB_USER:-}"

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "Set GITHUB_TOKEN (classic PAT with repo scope)." >&2
  exit 1
fi

if [ -z "$GITHUB_USER" ]; then
  GITHUB_USER="$(curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" \
    https://api.github.com/user | python3 -c 'import sys,json; print(json.load(sys.stdin)["login"])')"
fi

API="https://api.github.com/repos/${GITHUB_USER}/${REPO_NAME}"
if curl -fsSL -o /dev/null -H "Authorization: Bearer $GITHUB_TOKEN" "$API"; then
  echo "Repo already exists: https://github.com/${GITHUB_USER}/${REPO_NAME}"
else
  curl -fsSL -X POST -H "Authorization: Bearer $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/user/repos \
    -d "{\"name\":\"${REPO_NAME}\",\"description\":\"UR4Rec + Exp3RT + Avito Auto — shared MSc rerank project\",\"private\":false,\"auto_init\":false}" \
    >/dev/null
  echo "Created public repo: https://github.com/${GITHUB_USER}/${REPO_NAME}"
fi

git remote remove origin 2>/dev/null || true
git remote add origin "https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"
git push -u origin main
git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo "Done: https://github.com/${GITHUB_USER}/${REPO_NAME}"
