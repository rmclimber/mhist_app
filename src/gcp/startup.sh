#!/bin/bash

docker run \
    -e WANDB_API_KEY= ## \
    your-


gcloud compute instances create-with-container my-mhist-instance \
  --container-image=gcr.io/your-project-id/your-image-name:latest \
  --container-env=WANDB_API_KEY=your_real_api_key \
  --zone=us-central1-a \
  --scopes=https://www.googleapis.com/auth/cloud-platform