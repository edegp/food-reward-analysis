# Useful Commands

## Local Development
```bash
# diagnosis-chat-api
cd diagnosis-chat-api
uv sync
uv run uvicorn src.main:app --reload --port 8080

# diagnosis-summary-api
cd diagnosis-summary-api
uv sync
uv run uvicorn src.app:app --reload --port 8081

# Frontend
cd frontend
flutter run -d chrome
```

## Deployment
```bash
# Terraform
cd terraform
terraform plan
terraform apply

# Manual Cloud Run deploy
gcloud run deploy mbti-diagnosis-api --region asia-southeast1

# Check logs
gcloud run services logs read mbti-diagnosis-api --region asia-southeast1 --limit=50
```

## Secret Management
```bash
# Update secret (no newline!)
echo -n "VALUE" | gcloud secrets versions add SECRET_NAME --project=chat-mbti-458210 --data-file=-

# View secret
gcloud secrets versions access latest --secret=SECRET_NAME --project=chat-mbti-458210
```
