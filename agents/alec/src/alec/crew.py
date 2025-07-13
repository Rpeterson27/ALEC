"""
ALEC Language Learning Crew

This module implements a multi-agent system for language learning using CrewAI.

INTERACTION STRUCTURE:
- Conversational Agent (Alec): ONLY agent that interacts with users, presents all messages
- Curriculum Agent: Generates structured curriculum and tracks progress  
- Pronunciation Coach Agent: Analyzes pronunciation attempts with top-K IPA probabilities

AGENT RESPONSIBILITIES:
1. Conversational Agent: User-facing chat interface, contextualizes all system outputs
2. Curriculum Agent: Designs lessons, tracks progress, determines next steps
3. Pronunciation Coach Agent: Compares user IPA to correct IPA using top-K probabilities

All interactions flow through the Conversational Agent to maintain consistent user experience.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import weave
import os
from crewai import LLM

@dataclass
class UserProfile:
    """User profile and progress tracking for shared knowledge"""
    username: str
    native_language: str
    target_language: str
    level: str
    current_lesson: int = 0
    total_attempts: int = 0
    successful_attempts: int = 0
    current_phrase_attempts: int = 0  # Track attempts for current phrase
    max_suggested_retries: int = 3  # Suggest moving on after 3 attempts
    
    @property
    def success_rate(self) -> float:
        """Calculate success percentage for progress tracking"""
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_attempts / self.total_attempts) * 100
    
    @property
    def should_offer_to_advance(self) -> bool:
        """Check if user should be offered option to move on after retries"""
        return self.current_phrase_attempts >= self.max_suggested_retries

@dataclass
class PronunciationData:
    """Structured data for pronunciation analysis"""
    target_phrase: str
    user_ipa: str
    topk_results: List[List[Tuple[str, float]]]
    
class AlecKnowledge:
    """Shared knowledge base accessible to all agents"""
    
    def __init__(self, user_profile: UserProfile):
        self.user_profile = user_profile
        self.current_phrase = None
        self.current_target_ipa = None
        self.lesson_history = []
        self.session_data = {}
    
    def update_progress(self, passed: bool) -> None:
        """Update user progress tracking"""
        self.user_profile.total_attempts += 1
        self.user_profile.current_phrase_attempts += 1
        if passed:
            self.user_profile.successful_attempts += 1
    
    def advance_lesson(self) -> None:
        """Move to next lesson and reset phrase attempts"""
        self.user_profile.current_lesson += 1
        self.user_profile.current_phrase_attempts = 0  # Reset for new phrase
        if self.current_phrase:
            self.lesson_history.append(self.current_phrase)
    
    def reset_phrase_attempts(self) -> None:
        """Reset attempts counter for new phrase"""
        self.user_profile.current_phrase_attempts = 0
    
    def get_context_vars(self) -> Dict[str, Any]:
        """Get context variables for task interpolation"""
        return {
            'username': self.user_profile.username,
            'native_language': self.user_profile.native_language,
            'target_language': self.user_profile.target_language,
            'level': self.user_profile.level,
            'current_lesson': self.user_profile.current_lesson,
            'success_rate': round(self.user_profile.success_rate, 1),
            'current_phrase_attempts': self.user_profile.current_phrase_attempts,
            'should_offer_to_advance': self.user_profile.should_offer_to_advance
        }

@CrewBase
class Alec():
    """ALEC Language Learning Crew - Enterprise deployment ready"""

    def __init__(self, user_profile: UserProfile, enable_weave: bool = True):
        """
        Initialize ALEC crew with user profile for shared knowledge.
        
        Args:
            user_profile: User's learning profile and progress data
            enable_weave: Enable Weights & Biases Weave tracking (default: True)
        """
        super().__init__()
        self.knowledge = AlecKnowledge(user_profile)
        self.logger = logging.getLogger(__name__)
        self.weave_enabled = enable_weave
        
        # Configure Gemini LLM
        self.llm = self._configure_gemini_llm()
        
        # Initialize Weave tracking if enabled
        if self.weave_enabled:
            self._init_weave_tracking()
    
    def _configure_gemini_llm(self) -> LLM:
        """
        Configure Google Gemini Flash model for all agents.
        
        Returns:
            LLM: Configured Gemini model instance
        """
        try:
            # Get configuration from environment
            model = os.getenv('MODEL', 'gemini/gemini-2.5-flash-preview-04-17')
            api_key = os.getenv('GEMINI_API_KEY')
            
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            # Configure Gemini LLM with CrewAI
            llm = LLM(
                model=model,
                api_key=api_key,
                temperature=0.7,  # Balanced creativity for language learning
                max_tokens=1500,  # Adequate for educational responses
                top_p=0.9,        # Good diversity for language explanations
            )
            
            self.logger.info(f"Configured Gemini model: {model}")
            return llm
            
        except Exception as e:
            self.logger.error(f"Failed to configure Gemini LLM: {e}")
            # Fallback configuration
            return LLM(
                model="gemini/gemini-2.5-flash-preview-04-17",
                temperature=0.7
            )
    
    def _init_weave_tracking(self) -> None:
        """
        Initialize Weights & Biases Weave tracking for the ALEC crew.
        
        Sets up comprehensive tracking of:
        - Agent interactions and communications
        - Task executions and results
        - LLM calls with metadata and token usage
        - Tool usage and results
        - Custom metrics for language learning progress
        """
        try:
            # Get project name from environment or use default
            project_name = os.getenv('WEAVE_PROJECT_NAME', 'alec-language-learning')
            
            # Initialize Weave with project name
            weave.init(project_name=project_name)
            
            # Log initialization with user context
            weave.init_logger()
            self.logger.info(f"Weave tracking initialized for project: {project_name}")
            self.logger.info(f"Tracking user: {self.knowledge.user_profile.username} "
                           f"({self.knowledge.user_profile.native_language} -> "
                           f"{self.knowledge.user_profile.target_language})")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Weave tracking: {e}")
            self.weave_enabled = False
    
    @weave.op()
    def track_learning_session(self, session_type: str, result: str, 
                              metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Track learning session events with Weave.
        
        Args:
            session_type: Type of session (welcome, lesson, pronunciation, feedback)
            result: The output/result from the session
            metadata: Additional tracking metadata
            
        Returns:
            Dict containing session tracking data
        """
        tracking_data = {
            'session_type': session_type,
            'username': self.knowledge.user_profile.username,
            'target_language': self.knowledge.user_profile.target_language,
            'level': self.knowledge.user_profile.level,
            'current_lesson': self.knowledge.user_profile.current_lesson,
            'success_rate': self.knowledge.user_profile.success_rate,
            'current_phrase_attempts': self.knowledge.user_profile.current_phrase_attempts,
            'result_length': len(result),
            'result_preview': result[:100] + "..." if len(result) > 100 else result,
            'timestamp': weave.timestamp()
        }
        
        if metadata:
            tracking_data.update(metadata)
        
        return tracking_data
    
    @weave.op()
    def track_pronunciation_analysis(self, target_phrase: str, user_ipa: str,
                                   topk_results: List[List[Tuple[str, float]]],
                                   analysis_result: str, passed: bool) -> Dict[str, Any]:
        """
        Track pronunciation analysis with detailed metrics.
        
        Args:
            target_phrase: The target phrase being practiced
            user_ipa: User's IPA transcription
            topk_results: Top-K probability results from Allosaurus
            analysis_result: Feedback from pronunciation coach
            passed: Whether the attempt passed
            
        Returns:
            Dict containing pronunciation tracking data
        """
        # Calculate confidence metrics from top-K results
        avg_confidence = 0.0
        if topk_results:
            confidences = [max(phones, key=lambda x: x[1])[1] for phones in topk_results if phones]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        tracking_data = {
            'target_phrase': target_phrase,
            'user_ipa': user_ipa,
            'ipa_length': len(user_ipa),
            'topk_frames': len(topk_results),
            'avg_confidence': avg_confidence,
            'passed': passed,
            'attempt_number': self.knowledge.user_profile.current_phrase_attempts,
            'analysis_length': len(analysis_result),
            'feedback_type': 'pass' if passed else 'retry',
            'native_language': self.knowledge.user_profile.native_language,
            'user_level': self.knowledge.user_profile.level
        }
        
        return tracking_data

    @agent
    def conversational_agent(self) -> Agent:
        """
        User-facing conversational agent (Alec) that handles ALL user interactions.
        
        This is the ONLY agent that communicates directly with users.
        Receives structured data from other agents and converts it into
        friendly, encouraging chat messages. Manages progression decisions.
        """
        return Agent(
            config=self.agents_config['conversational_agent'], # type: ignore[index]
            llm=self.llm,  # Use Gemini Flash
            verbose=True,
            allow_delegation=False,
            memory=True
        )

    @agent
    def curriculum_agent(self) -> Agent:
        """
        Domain expert that generates and manages structured curriculum.
        
        Tracks user progress and determines next lessons based on performance.
        Never interacts with users - sends curriculum to Conversational Agent.
        """
        return Agent(
            config=self.agents_config['curriculum_agent'], # type: ignore[index]
            llm=self.llm,  # Use Gemini Flash
            verbose=True,
            allow_delegation=False,
            memory=True
        )

    @agent
    def pronunciation_coach_agent(self) -> Agent:
        """
        Domain expert that analyzes pronunciation using IPA comparison and top-K probabilities.
        
        Receives target phrase, user IPA string, and top-K per-phone probability results.
        Uses advanced analysis to provide nuanced feedback and pass/fail determination.
        """
        return Agent(
            config=self.agents_config['pronunciation_coach_agent'], # type: ignore[index]
            llm=self.llm,  # Use Gemini Flash
            verbose=True,
            allow_delegation=False,
            memory=True
        )

    @task
    def welcome_user_task(self) -> Task:
        """Task for Conversational Agent to welcome user to learning session"""
        return Task(
            config=self.tasks_config['welcome_user_task'], # type: ignore[index]
        )

    @task
    def get_next_lesson_task(self) -> Task:
        """Task for Curriculum Agent to determine next lesson/phrase"""
        return Task(
            config=self.tasks_config['get_next_lesson_task'], # type: ignore[index]
        )

    @task
    def analyze_pronunciation_task(self) -> Task:
        """Task for Pronunciation Coach to analyze user's pronunciation attempt"""
        return Task(
            config=self.tasks_config['analyze_pronunciation_task'], # type: ignore[index]
        )

    @task
    def present_lesson_task(self) -> Task:
        """Task for Conversational Agent to present curriculum to user"""
        return Task(
            config=self.tasks_config['present_lesson_task'], # type: ignore[index]
        )

    @task
    def present_feedback_task(self) -> Task:
        """Task for Conversational Agent to present pronunciation feedback to user"""
        return Task(
            config=self.tasks_config['present_feedback_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ALEC language learning crew for Enterprise deployment"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            memory=True,
            # Use Gemini for all LLM operations
            manager_llm=self.llm,
            # Use Google's embedding model for consistency
            embedder={
                "provider": "google",
                "config": {
                    "model": "models/text-embedding-004",
                    "api_key": os.getenv('GEMINI_API_KEY')
                }
            }
        )

    def start_new_session(self) -> str:
        """
        Initialize a new learning session for the user.
        Only exposes Conversational Agent interface.
        
        Returns:
            str: Welcome message from Conversational Agent (Alec)
        """
        try:
            # Create welcome task with user context
            welcome_task = Task(
                description=self.tasks_config['welcome_user_task']['description'].format(
                    **self.knowledge.get_context_vars()
                ),
                expected_output=self.tasks_config['welcome_user_task']['expected_output'].format(
                    **self.knowledge.get_context_vars()
                ),
                agent=self.conversational_agent()
            )
            
            result = self.conversational_agent().execute_task(welcome_task)
            
            # Track session start with Weave
            if self.weave_enabled:
                self.track_learning_session(
                    session_type='welcome',
                    result=result,
                    metadata={'session_start': True}
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error starting session: {e}")
            fallback_result = f"Hello {self.knowledge.user_profile.username}! I'm Alec, ready to help you learn {self.knowledge.user_profile.target_language}!"
            
            # Track fallback with Weave
            if self.weave_enabled:
                self.track_learning_session(
                    session_type='welcome_fallback',
                    result=fallback_result,
                    metadata={'error': str(e)}
                )
            
            return fallback_result

    def get_next_curriculum_step(self) -> str:
        """
        Get next curriculum step through Conversational Agent only.
        
        Returns:
            str: User-friendly message with next lesson from Alec
        """
        try:
            context_vars = self.knowledge.get_context_vars()
            
            # Curriculum agent generates lesson
            curriculum_task = Task(
                description=self.tasks_config['get_next_lesson_task']['description'].format(**context_vars),
                expected_output=self.tasks_config['get_next_lesson_task']['expected_output'].format(**context_vars),
                agent=self.curriculum_agent()
            )
            
            curriculum_result = self.curriculum_agent().execute_task(curriculum_task)
            
            # Conversational agent presents to user
            presentation_vars = {**context_vars, 'curriculum_result': curriculum_result}
            presentation_task = Task(
                description=self.tasks_config['present_lesson_task']['description'].format(**presentation_vars),
                expected_output=self.tasks_config['present_lesson_task']['expected_output'].format(**presentation_vars),
                agent=self.conversational_agent()
            )
            
            result = self.conversational_agent().execute_task(presentation_task)
            
            # Track lesson delivery with Weave
            if self.weave_enabled:
                self.track_learning_session(
                    session_type='lesson',
                    result=result,
                    metadata={
                        'curriculum_content': curriculum_result[:200] + "..." if len(curriculum_result) > 200 else curriculum_result,
                        'lesson_number': self.knowledge.user_profile.current_lesson
                    }
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting curriculum: {e}")
            fallback_result = "Let's try a new phrase! I'll give you something to practice."
            
            # Track fallback with Weave
            if self.weave_enabled:
                self.track_learning_session(
                    session_type='lesson_fallback',
                    result=fallback_result,
                    metadata={'error': str(e)}
                )
            
            return fallback_result

    def analyze_pronunciation_attempt(self, target_phrase: str, user_ipa: str, 
                                    topk_results: List[List[Tuple[str, float]]]) -> str:
        """
        Analyze pronunciation attempt using top-K IPA probabilities.
        Only returns response from Conversational Agent.
        
        Args:
            target_phrase: The phrase user was trying to pronounce
            user_ipa: Flat IPA string from Allosaurus (no word boundaries)
            topk_results: Per-frame list of (phone, probability) tuples
                         e.g., [ [('k',0.7),('h',0.2)], [('o',0.38),('ɔ',0.38)] ]
            
        Returns:
            str: User-friendly feedback message from Alec
        """
        try:
            context_vars = self.knowledge.get_context_vars()
            analysis_vars = {
                **context_vars,
                'target_phrase': target_phrase,
                'user_ipa': user_ipa,
                'topk_results': topk_results
            }
            
            # Pronunciation coach analyzes with top-K data
            analysis_task = Task(
                description=self.tasks_config['analyze_pronunciation_task']['description'].format(**analysis_vars),
                expected_output=self.tasks_config['analyze_pronunciation_task']['expected_output'].format(**analysis_vars),
                agent=self.pronunciation_coach_agent()
            )
            
            analysis_result = self.pronunciation_coach_agent().execute_task(analysis_task)
            
            # Conversational agent presents feedback
            feedback_vars = {**context_vars, 'analysis_result': analysis_result}
            feedback_task = Task(
                description=self.tasks_config['present_feedback_task']['description'].format(**feedback_vars),
                expected_output=self.tasks_config['present_feedback_task']['expected_output'].format(**feedback_vars),
                agent=self.conversational_agent()
            )
            
            result = self.conversational_agent().execute_task(feedback_task)
            
            # Update progress tracking
            passed = any(word in analysis_result.lower() for word in ['pass', 'correct', 'good', 'excellent'])
            self.knowledge.update_progress(passed)
            
            # Track pronunciation analysis with Weave
            if self.weave_enabled:
                self.track_pronunciation_analysis(
                    target_phrase=target_phrase,
                    user_ipa=user_ipa,
                    topk_results=topk_results,
                    analysis_result=analysis_result,
                    passed=passed
                )
                
                # Track the user-facing result too
                self.track_learning_session(
                    session_type='pronunciation_feedback',
                    result=result,
                    metadata={
                        'target_phrase': target_phrase,
                        'passed': passed,
                        'attempt_number': self.knowledge.user_profile.current_phrase_attempts,
                        'should_offer_advance': self.knowledge.user_profile.should_offer_to_advance
                    }
                )
            
            if passed:
                self.knowledge.advance_lesson()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing pronunciation: {e}")
            fallback_result = "Good effort! Let's try that phrase again, focusing on the pronunciation."
            
            # Track fallback with Weave
            if self.weave_enabled:
                self.track_learning_session(
                    session_type='pronunciation_error',
                    result=fallback_result,
                    metadata={
                        'error': str(e),
                        'target_phrase': target_phrase,
                        'user_ipa': user_ipa
                    }
                )
            
            return fallback_result
