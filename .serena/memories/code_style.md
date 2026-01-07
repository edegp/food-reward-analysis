# Code Style Guidelines

## Python (Backend)
- Package manager: uv
- Formatting: ruff
- Type hints: Required
- Async: Use async/await for I/O operations

## Dart (Frontend)
- Flutter conventions
- State management: Provider/Riverpod

## Commit Messages
- Conventional commits (feat:, fix:, chore:, etc.)
- Japanese descriptions acceptable

## Environment Variables
- Use os.getenv() with defaults for optional vars
- Secrets via GCP Secret Manager
- Never commit secrets to git
