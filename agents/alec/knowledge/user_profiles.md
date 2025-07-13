# ALEC User Profile Knowledge Base

This knowledge base contains information about user profiles and learning progress for the ALEC language learning system.

## User Profile Structure

Each user has the following profile data:

- **username**: Unique identifier for personalization
- **native_language**: User's native language (affects pronunciation coaching)
- **target_language**: Language being learned
- **level**: beginner, intermediate, or advanced
- **current_lesson**: Current lesson number in the curriculum
- **total_attempts**: Total pronunciation attempts made
- **successful_attempts**: Number of successful pronunciation attempts
- **success_rate**: Calculated percentage of successful attempts

## Learning Progress Tracking

The system tracks:

1. **Lesson Progression**: Sequential advancement through curriculum
2. **Pronunciation Accuracy**: Pass/fail status for each attempt
3. **Success Metrics**: Overall performance statistics
4. **Session History**: Record of completed lessons and phrases

## Shared Knowledge Access

All three agents have access to this shared knowledge:

- **Conversational Agent**: Uses profile for personalization and progression decisions
- **Curriculum Agent**: Adapts lesson difficulty based on success rate and level
- **Pronunciation Coach**: Considers native language for targeted feedback

## Enterprise Deployment Notes

For Crew.ai Enterprise deployment:

- User profiles are maintained per session
- Progress data is updated in real-time
- Knowledge is shared across all agents in the crew
- Supports concurrent user sessions with isolated contexts