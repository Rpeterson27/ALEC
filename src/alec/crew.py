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

    def __init__(self, user_profile: UserProfile):
        """
        Initialize ALEC crew with user profile for shared knowledge.
        
        Args:
            user_profile: User's learning profile and progress data
        """
        super().__init__()
        self.knowledge = AlecKnowledge(user_profile)
        self.logger = logging.getLogger(__name__)
        
        # Configure Gemini LLM
        self.llm = self._configure_gemini_llm()
    
    def _configure_gemini_llm(self) -> LLM:
        """
        Configure Google Gemini Flash model for all agents.
        
        Returns:
            LLM: Configured Gemini model instance
        """
        try:
            # Get configuration from environment
            model = os.getenv('MODEL', 'gemini/gemini-2.5-flash')
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
                model="gemini/gemini-2.5-flash",
                temperature=0.7
            )
    

    @agent
    def conversational_agent(self) -> Agent:
        """
        User-facing conversational agent (Alec) that handles ALL user interactions.
        
        This is the ONLY agent that communicates directly with users.
        Receives structured data from other agents and converts it into
        friendly, encouraging chat messages. Manages progression decisions.
        """
        return Agent(
            # config=self.agents_config['conversational_agent'], # type: ignore[index]
            role="User-Facing Language Learning Coach for {username}",
            goal="Provide friendly, encouraging interaction with {username} learning {target_language}, presenting curriculum steps and pronunciation feedback in a supportive chat format",
            backstory="You are a patient, encouraging {target_language} language tutor named Alec who specializes in making language learning feel like a natural conversation with {username}. You receive structured curriculum and feedback from expert systems and transform them into warm, personalized messages that keep learners motivated and engaged. You know when to celebrate progress, when to offer gentle correction, and when to move forward with new material. You are the ONLY agent that communicates directly with {username}, and you manage all progression decisions including when to retry or advance.",
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
            # config=self.agents_config['curriculum_agent'], # type: ignore[index]
            llm=self.llm,  # Use Gemini Flash
            role="{target_language} Language Curriculum Expert and Progress Tracker",
            goal="Design structured {target_language} curriculum for {username} ({level} level, {native_language} speaker) and track learning progress",
            backstory="You are an experienced {target_language} curriculum designer with deep pedagogical knowledge. You create progressive lesson sequences that avoid proper nouns, starting from basic pronunciation and building systematically. You understand how to sequence lessons for maximum retention for {native_language} speakers learning {target_language}. You track {username}'s progress and adapt the pace based on their mastery of previous concepts, ensuring they're always challenged but never overwhelmed. You never interact with users directly - you send curriculum steps to the Conversational Agent.",
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
            role="{target_language} Pronunciation Analysis and Feedback Specialist",
            goal="Analyze {username}'s pronunciation attempts using IPA comparison and top-K probability analysis to provide detailed, actionable feedback for {target_language} improvement",
            backstory="""You are a phonetics expert specializing in {target_language} pronunciation coaching with advanced speech analysis capabilities. You analyze IPA transcriptions and top-K phone probabilities from speech recognition systems, comparing against correct pronunciation patterns. You can infer likely word/phoneme boundaries from flat IPA strings and understand common pronunciation challenges for {native_language} speakers learning {target_language}. You provide nuanced feedback like "You nearly said this sound but may have confused it with..." using probability data. You never interact with users directly - you return structured results to the Conversational Agent.""",
            llm=self.llm,  # Use Gemini Flash
            verbose=True,
            allow_delegation=False,
            memory=True
        )

    @task
    def welcome_user_task(self) -> Task:
        """Task for Conversational Agent to welcome user to learning session"""
        return Task(
            description="Welcome {username} to their {target_language} learning session. They are at {level} level and speak {native_language} natively. Create a warm, encouraging welcome message that sets a positive tone for learning and asks if they're ready to begin or continue their studies.",
            expected_output="A personalized, encouraging welcome message that makes {username} feel comfortable and motivated to start learning {target_language}.",
            agent=self.conversational_agent()
        )

    @task
    def get_next_lesson_task(self) -> Task:
        """Task for Curriculum Agent to determine next lesson/phrase"""
        return Task(
            description="Based on {username}'s current progress (lesson {current_lesson}, {success_rate}% success rate), generate the next appropriate {target_language} phrase for a {level} level {native_language} speaker. Avoid proper nouns. Include the target phrase, correct IPA transcription, difficulty level, and learning objectives. Consider their native language background for targeted instruction.",
            expected_output="A structured lesson containing: target phrase (avoiding proper nouns), correct IPA transcription, difficulty level, learning objectives, and specific pronunciation focus areas for {native_language} speakers learning {target_language}.",
            agent=self.curriculum_agent()
        )

    @task
    def analyze_pronunciation_task(self) -> Task:
        """Task for Pronunciation Coach to analyze user's pronunciation attempt"""
        return Task(
            description="Analyze {username}'s pronunciation attempt using advanced IPA and probability analysis. Compare the user's IPA pronunciation against the correct IPA for the target phrase. Use available probability data to identify near-misses and confusion patterns. Consider their {native_language} background for targeted feedback. Provide nuanced analysis based on available data. Infer word boundaries from the IPA string if needed.",
            expected_output="Detailed pronunciation analysis with: pass/fail status, specific feedback, identified pronunciation issues, improvement suggestions, and insights explaining near-misses and alternative pronunciations detected.",
            agent=self.pronunciation_coach_agent()
        )

    @task
    def present_lesson_task(self) -> Task:
        """Task for Conversational Agent to present curriculum to user"""
        return Task(
            description="Present the curriculum step to {username} in a friendly, encouraging manner. Convert the structured lesson data into natural, supportive chat message format. Make it clear what they need to do and keep them motivated to practice the pronunciation.",
            expected_output="A friendly, conversational presentation of the lesson with clear instructions for {username}, formatted as natural dialogue that encourages practice and engagement.",
            agent=self.conversational_agent()
        )

    @task
    def present_feedback_task(self) -> Task:
        """Task for Conversational Agent to present pronunciation feedback to user"""
        return Task(
            description="Present pronunciation analysis to {username} in an encouraging way. Convert the technical feedback into supportive, actionable guidance. Based on their attempt history, decide whether they should retry, move on, or let them choose. If they've made multiple attempts, offer them the choice to move on to the next phrase while being encouraging about their effort. Maintain motivation and celebrate any progress made.",
            expected_output="Encouraging, personalized feedback message with clear next steps (retry, advance, or user choice), formatted as supportive conversation that keeps {username} motivated. If offering to advance after retries, phrase it positively like 'You've given this great effort! Would you like to try once more or move on to something new?'",
            agent=self.conversational_agent()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ALEC language learning crew for local development"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            memory=True,
            # Use Gemini for all LLM operations
            manager_llm=self.llm,
            # Use Google embedder to avoid OpenAI dependency
            embedder={
                "provider": "google",
                "config": {
                    "model": "models/text-embedding-004",
                    "api_key": os.getenv('GEMINI_API_KEY')
                }
            }
        )