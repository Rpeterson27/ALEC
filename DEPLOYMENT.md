# ALEC CrewAI Enterprise Deployment Guide

This guide walks through deploying the ALEC language learning crew to CrewAI Enterprise.

## Prerequisites

✅ **Code Repository**: ALEC crew code pushed to GitHub  
✅ **CrewAI Enterprise Account**: Access to enterprise.crewai.com  
✅ **Environment Variables**: Gemini API key ready  
✅ **Dependencies**: All required packages in pyproject.toml  

## Deployment Methods

### Option 1: Web Interface Deployment (Recommended)

1. **Login to CrewAI Enterprise**
   ```
   https://enterprise.crewai.com
   ```

2. **Connect GitHub Repository**
   - Click "Connect GitHub" 
   - Authorize CrewAI access
   - Select your repository
   - Choose the `crewai_integration` branch

3. **Configure Environment Variables**
   Set these **REQUIRED** variables:
   ```
   MODEL=gemini/gemini-2.5-flash-preview-04-17
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   
   **Optional** variables for Weave tracking:
   ```
   WANDB_API_KEY=your_wandb_api_key_here
   WEAVE_PROJECT_NAME=alec-production
   ```

4. **Deploy**
   - Click "Deploy" button
   - Monitor deployment progress (10-15 minutes)
   - Wait for deployment success confirmation

### Option 2: CLI Deployment

1. **Install CrewAI CLI**
   ```bash
   pip install crewai[tools]
   ```

2. **Login**
   ```bash
   crewai login
   ```

3. **Deploy from Repository**
   ```bash
   cd /path/to/alec
   crewai deploy create
   ```

4. **Set Environment Variables**
   ```bash
   crewai deploy env set MODEL=gemini/gemini-2.5-flash-preview-04-17
   crewai deploy env set GEMINI_API_KEY=your_api_key
   ```

5. **Monitor Status**
   ```bash
   crewai deploy status
   ```

## Environment Variables Configuration

### Required Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `MODEL` | `gemini/gemini-2.5-flash-preview-04-17` | Gemini Flash model |
| `GEMINI_API_KEY` | Your API key | Google AI Studio key |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_API_KEY` | None | Weights & Biases tracking |
| `WEAVE_PROJECT_NAME` | `alec-production` | Weave project name |
| `DEFAULT_TARGET_LANGUAGE` | `French` | Default learning language |
| `DEFAULT_NATIVE_LANGUAGE` | `English` | Default native language |
| `DEFAULT_LEVEL` | `beginner` | Default skill level |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `DISABLE_WEAVE` | `false` | Disable tracking |

## Post-Deployment

### 1. Verify Deployment
After deployment completes, you'll receive:
- ✅ Deployment URL
- ✅ API endpoint URLs
- ✅ Health check status

### 2. API Endpoints
Your deployed ALEC crew will expose these endpoints:

```
POST /sessions
GET /sessions/{username}/lesson  
POST /sessions/{username}/pronunciation
POST /sessions/{username}/skip
```

### 3. Test Deployment
```bash
# Health check
curl https://your-deployment-url.crewai.com/health

# Create session
curl -X POST https://your-deployment-url.crewai.com/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "username": "test_user",
    "native_language": "English", 
    "target_language": "French",
    "level": "beginner"
  }'
```

### 4. Monitor Performance
- **CrewAI Dashboard**: View execution metrics
- **Weave Dashboard**: Detailed AI performance tracking
- **Logs**: Application and error logs

## Troubleshooting

### Common Issues

1. **Environment Variable Errors**
   - Ensure `GEMINI_API_KEY` is set correctly
   - Check variable naming (no `_TOKEN` or `_SECRET` suffixes)

2. **Deployment Timeouts**
   - First deployment takes 10-15 minutes
   - Check deployment logs for specific errors

3. **API Errors**
   - Verify Gemini API key has sufficient quota
   - Check model name is exactly: `gemini/gemini-2.5-flash-preview-04-17`

### Support Channels
- **CrewAI Support**: enterprise-support@crewai.com
- **Documentation**: https://docs.crewai.com/enterprise
- **Status Page**: https://status.crewai.com

## Production Considerations

### Security
- ✅ Environment variables encrypted at rest
- ✅ API keys never logged or exposed
- ✅ HTTPS enforced for all endpoints

### Scaling
- ✅ Auto-scaling based on demand
- ✅ Load balancing across instances
- ✅ Geographic distribution available

### Monitoring
- ✅ Built-in health checks
- ✅ Performance metrics dashboard
- ✅ Error tracking and alerting
- ✅ Weave integration for AI observability

## Success Criteria

After deployment, verify these work:
- [ ] Health endpoint returns 200 OK
- [ ] Session creation works with valid user data
- [ ] Lesson generation returns Gemini-powered responses
- [ ] Pronunciation analysis processes IPA data
- [ ] Weave tracking captures all interactions (if enabled)
- [ ] Error handling gracefully manages failures

Your ALEC language learning crew is now ready for production use! 🚀