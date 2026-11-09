#!/usr/bin/env bash
# ==============================================================================
# Script to upload Federated SAM project to GitHub (without images or weights)
# ==============================================================================
set -e

REPO_URL="$1"

if [ -z "$REPO_URL" ]; then
    echo "======================================================================"
    echo "Usage: ./upload_to_github.sh <YOUR_GITHUB_REPO_URL>"
    echo ""
    echo "Example:"
    echo "  ./upload_to_github.sh https://github.com/your-username/fed-sam-polyp.git"
    echo "======================================================================"
    exit 1
fi

echo ">>> Target GitHub repository: $REPO_URL"

# Ensure we are in the project root
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 1. Update remote origin
if git remote | grep -q "origin"; then
    echo ">>> Updating remote origin to: $REPO_URL"
    git remote set-url origin "$REPO_URL"
else
    echo ">>> Adding remote origin: $REPO_URL"
    git remote add origin "$REPO_URL"
fi

# 2. Rename branch to main if needed
git branch -M main

# 3. Stage all code, scripts, configs, and reports (respecting .gitignore)
echo ">>> Staging project files..."
git add -A

# 4. Commit changes
echo ">>> Creating commit..."
git commit -m "feat: Federated SAM for domain-generalized polyp segmentation (Fed-FGS, MoDE, Scale-MoE)" || echo "No new changes to commit."

# 5. Push to GitHub
echo ">>> Pushing to $REPO_URL (main branch)..."
git push -u origin main

echo ""
echo "======================================================================"
echo " Successfully uploaded to GitHub!"
echo "======================================================================"
