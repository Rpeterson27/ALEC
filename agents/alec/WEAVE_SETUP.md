# ALEC + Weights & Biases Weave Integration

This document explains how to set up and use Weights & Biases Weave with the ALEC language learning crew for comprehensive tracking and monitoring.

## What is Weave?

Weave is W&B's toolkit for tracking, experimenting with, evaluating, and improving AI applications. For ALEC, it provides:

- **Automatic Tracing**: All agent interactions, task executions, and LLM calls
- **Performance Monitoring**: Success rates, response times, and learning progress
- **Debugging**: Detailed trace analysis for troubleshooting
- **Evaluation**: Custom metrics for language learning effectiveness
- **Collaboration**: Shared dashboards for team insights

## Quick Setup

### 1. Install Dependencies

```bash
pip install weave wandb
```

Or install with the ALEC project:
```bash
pip install -e .
```

### 2. Get Your W&B API Key

1. Sign up at [wandb.ai](https://wandb.ai)
2. Go to your profile settings
3. Copy your API key

### 3. Configure Environment

Create a `.env` file (copy from `.env.example`):

```bash
# Required for Weave tracking
WANDB_API_KEY=your_wandb_api_key_here
WEAVE_PROJECT_NAME=alec-language-learning

# Required for ALEC functionality with Gemini Flash
MODEL=gemini/gemini-2.5-flash-preview-04-17
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run ALEC with Weave

```python
from alec.crew import Alec, UserProfile

# Create user profile
user_profile = UserProfile(
    username="Alice",
    native_language="English", 
    target_language="French",
    level="beginner"
)

# Initialize with Weave tracking enabled (default)
alec_crew = Alec(user_profile, enable_weave=True)

# Use normally - all interactions are automatically tracked
welcome = alec_crew.start_new_session()
lesson = alec_crew.get_next_curriculum_step()
```

## What Gets Tracked

### Automatic CrewAI Tracking
Weave automatically captures:
- Agent communications and task executions
- LLM calls with token usage and costs
- Tool usage and results
- Execution times and performance metrics

### Custom ALEC Metrics
We've added specialized tracking for:

#### Learning Sessions
- Session type (welcome, lesson, pronunciation, feedback)
- User progress (success rate, current lesson, attempts)
- Response quality and length
- Error handling and fallbacks

#### Pronunciation Analysis
- Target phrases and user IPA attempts
- Top-K probability data from Allosaurus
- Confidence scores and analysis results
- Pass/fail status and retry patterns
- Native language impact on pronunciation

### Tracked Data Examples

```python
# Session tracking includes:
{
    'session_type': 'pronunciation_feedback',
    'username': 'Alice',
    'target_language': 'French',
    'success_rate': 67.5,
    'current_phrase_attempts': 2,
    'passed': True,
    'result_preview': 'Great job! Your pronunciation of...'
}

# Pronunciation analysis includes:
{
    'target_phrase': 'bonjour',
    'user_ipa': 'bonʒuʁ',
    'avg_confidence': 0.74,
    'attempt_number': 2,
    'passed': True,
    'native_language': 'English'
}
```

## Viewing Your Data

### Weave Dashboard
Access your tracking data at:
```
https://wandb.ai/YOUR_USERNAME/alec-language-learning/weave
```

### Key Views
1. **Traces**: See complete interaction flows
2. **Operations**: Monitor individual agent calls
3. **Objects**: Track user profiles and session data
4. **Evaluations**: Performance metrics and trends

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WANDB_API_KEY` | Your W&B API key | Required |
| `WEAVE_PROJECT_NAME` | Project name in W&B | `alec-language-learning` |
| `DISABLE_WEAVE` | Disable tracking | `false` |

### Code Configuration

```python
# Disable tracking for a specific crew
alec_crew = Alec(user_profile, enable_weave=False)

# Custom project name
import os
os.environ['WEAVE_PROJECT_NAME'] = 'my-custom-project'
alec_crew = Alec(user_profile)
```

## API Integration

The FastAPI endpoints automatically include Weave tracking:

```python
from alec.main import AlecAPI

# API with tracking enabled
api = AlecAPI(enable_weave=True)

# All endpoint calls are automatically tracked
welcome = api.create_user_session("Alice", "English", "French", "beginner")
lesson = api.get_next_lesson("Alice")
feedback = api.submit_pronunciation("Alice", "audio.wav")
```

## Enterprise Deployment

For CrewAI Enterprise deployment, ensure these environment variables are set:

```bash
# In your deployment configuration
WANDB_API_KEY=your_api_key
WEAVE_PROJECT_NAME=alec-production
OPENAI_API_KEY=your_openai_key
```

Weave tracking will automatically work in production with no code changes.

## Troubleshooting

### Common Issues

1. **"Weave tracking disabled"**
   - Check that `WANDB_API_KEY` is set
   - Verify API key is valid at wandb.ai

2. **No traces appearing**
   - Confirm network access to wandb.ai
   - Check project name matches your dashboard

3. **Performance concerns**
   - Weave adds minimal overhead (~1-2ms per call)
   - Disable with `DISABLE_WEAVE=true` if needed

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

1. **Project Organization**: Use descriptive project names for different environments
2. **Data Privacy**: Weave respects your data privacy - see W&B privacy policy
3. **Performance**: Keep tracking enabled in production for valuable insights
4. **Team Collaboration**: Share project access for team visibility
5. **Custom Metrics**: Add domain-specific tracking with `@weave.op()`

## Support

- **Weave Documentation**: [weave-docs.wandb.ai](https://weave-docs.wandb.ai)
- **CrewAI Integration**: [docs.crewai.com/observability/weave](https://docs.crewai.com/observability/weave)
- **ALEC Issues**: Create an issue in this repository