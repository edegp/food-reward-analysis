# Task Completion Checklist

## Before Deploying
1. Run local tests
2. Check for hardcoded secrets
3. Verify environment variables
4. Test database connections

## After Code Changes
1. Commit with conventional commit message
2. Push to trigger Cloud Build
3. Monitor Cloud Run logs for errors
4. Verify functionality in production

## Common Issues & Solutions

### API Key Errors (plugin_credentials.cc)
- Secret Managerのキーに改行が含まれている
- 解決: `echo -n "KEY" | gcloud secrets versions add ...`

### KeyError for Environment Variables
- Cloud Runが古いイメージを使用している
- 解決: 新しいリビジョンをデプロイ

### Database Connection Issues
- DATABASE_URL環境変数を確認
- Supabase接続プーラーのURLを使用
