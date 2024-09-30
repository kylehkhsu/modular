#!/bin/bash

# Define the base directory where your range of directories are located
DIR="/iris/u/kylehsu/runs/modular/2024-09-25/09-52-35"

for INT in {30..49}; do
  # Navigate into each directory's wandb/latest-run
  if [ -d "$DIR/$INT/wandb/latest-run" ]; then
    cd "$DIR/$INT/wandb/latest-run" || continue

    # Print the directory being synced
    echo "Syncing directory: $DIR/$INT/wandb/latest-run"

    # Execute "wandb sync ."
    wandb sync -e iris_viscam -p modular --id "arxiv_qlae_$INT" . &

    # Return to the base directory
    cd "$DIR" || exit
  else
    echo "Directory $DIR/$INT/wandb/latest-run does not exist, skipping..."
  fi
done