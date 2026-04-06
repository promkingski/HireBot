# Mission Log Analyzer – Secure Agentic Platform on GCP
This project demonstrates how to build a secure, scalable AI‑enabled platform on
Google Cloud Platform for regulated environments. It includes a simple web
application that allows you to upload mission logs and returns a summary of each
file using Vertex AI (or a local fallback summariser). The project is designed
to show how an experienced Azure/AWS engineer can rapidly ramp up on GCP and
apply best‑practice patterns.
## Architecture
- Frontend and API are containerised and deployed on Cloud Run.
- File uploads are stored in Cloud Storage.
- Summarisation uses Vertex AI’s text generation model (or a fallback).
- Logging and monitoring are handled by Cloud Logging and Cloud Monitoring.
## Deployment
Follow these steps to deploy the project:
1. Enable required services: run, compute, artifactregistry, aiplatform,
secretmanager.
2. Create a Cloud Storage bucket and an Artifact Registry repository.
3. Build and deploy using Cloud Build (`gcloud builds submit --config
cloudbuild.yaml`).
4. Visit the deployed Cloud Run URL and test file uploads.
## Security
This demo enforces least‑privilege IAM and uses Secret Manager for sensitive
configuration. For regulated workloads, consider adding VPC Service Controls
and CMEK encryption.
