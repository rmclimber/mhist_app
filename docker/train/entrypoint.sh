#!/bin/bash
set -e

# variables
BRANCH=${GIT_BRANCH:-""}  # Empty = stick with default branch

# check whether GITHUB_TOKEN is set
if [ -z "$GITHUB_API_KEY" ]; then
    echo "GITHUB_API_KEY not set!"
    exit 1
fi

# Optionally pull config from GCS
gsutil cp gs://mhist-configs/config.yml /workspace/config.yml

# Clone the private repo (adjust the URL and branch if needed)
git clone https://$GITHUB_API_KEY@github.com/rmclimber/mhist_app.git /workspace/mhist_app

# Change directory to repo
cd /workspace/mhist_app

# Optionally switch to a different branch
if [ -n "$BRANCH" ]; then
    echo "🔀 Switching to branch: $BRANCH"
    git fetch origin "$BRANCH"
    git checkout "$BRANCH"
fi

# Now run your training script
echo "Beginning training..."
python -m src.experiment.model_training --config /workspace/config.yml
echo "Training complete"