#!/usr/bin/env python
"""
ALEC Language Learning API

FastAPI endpoints that integrate with the ALEC crew system.
Only the Conversational Agent interface is exposed to maintain clean user interaction.

API Endpoints:
- POST /sessions: Create new learning session
- GET /sessions/{username}/lesson: Get next curriculum step  
- POST /sessions/{username}/pronunciation: Analyze pronunciation attempt

All responses come through the Conversational Agent (Alec) to ensure consistent,
encouraging user experience.
"""

import sys
import warnings
import logging
import weave
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv

from alec.crew import Alec, UserProfile

# Load environment variables from .env file
load_dotenv()

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlecAPI:
    """
    API layer for ALEC language learning crew.
    Integrates with FastAPI backend for Enterprise deployment.
    
    All interactions flow through the Conversational Agent to maintain
    consistent user experience.
    """
    
    def __init__(self, enable_weave: bool = True):
        self.active_crews: Dict[str, Alec] = {}
        self.logger = logging.getLogger(__name__)
        self.weave_enabled = enable_weave
        
        # Initialize Weave for API-level tracking
        if self.weave_enabled:
            self._init_api_weave_tracking()
    
    def _init_api_weave_tracking(self) -> None:
        """Initialize Weave tracking for the API layer"""
        try:
            # Weave is already initialized by individual crews, 
            # but we can add API-level logging here
            self.logger.info("ALEC API initialized with Weave tracking enabled")
        except Exception as e:
            self.logger.warning(f"API Weave tracking setup warning: {e}")
    
    def create_user_session(self, username: str, native_language: str, 
                          target_language: str, level: str) -> str:
        """
        Create new learning session - API endpoint.
        Only exposes Conversational Agent (Alec) interface.
        
        Args:
            username: User's name for personalization
            native_language: User's native language 
            target_language: Language they want to learn
            level: beginner, intermediate, advanced
            
        Returns:
            str: Welcome message from Alec
        """
        try:
            user_profile = UserProfile(
                username=username,
                native_language=native_language,
                target_language=target_language,
                level=level
            )
            
            crew = Alec(user_profile, enable_weave=self.weave_enabled)
            self.active_crews[username] = crew
            
            welcome_message = crew.start_new_session()
            self.logger.info(f"Created session for {username}: {native_language} -> {target_language} ({level})")
            
            return welcome_message
            
        except Exception as e:
            self.logger.error(f"Error creating session for {username}: {e}")
            return f"Welcome {username}! I'm Alec, ready to help you learn {target_language}."
    
    def skip_to_next_lesson(self, username: str) -> str:
        """
        Skip current phrase and move to next lesson - API endpoint.
        Allows users to progress when struggling with current phrase.
        
        Args:
            username: User identifier
            
        Returns:
            str: Encouraging message about moving on from Alec
        """
        if username not in self.active_crews:
            return "Session not found. Please start a new session."
        
        try:
            # Advance to next lesson and reset attempts
            self.active_crews[username].knowledge.advance_lesson()
            
            # Get next lesson
            lesson_message = self.active_crews[username].get_next_curriculum_step()
            self.logger.info(f"Skipped to next lesson for {username}")
            return f"No problem! Let's try something new. {lesson_message}"
            
        except Exception as e:
            self.logger.error(f"Error skipping lesson for {username}: {e}")
            return "No worries! Let's move on to something new you can practice."

    def get_next_lesson(self, username: str) -> str:
        """
        Get next curriculum step - API endpoint.
        Only returns response from Conversational Agent.
        
        Args:
            username: User identifier
            
        Returns:
            str: Next lesson presented by Alec
        """
        if username not in self.active_crews:
            return "Session not found. Please start a new session."
        
        try:
            # Reset attempts for new lesson
            self.active_crews[username].knowledge.reset_phrase_attempts()
            lesson_message = self.active_crews[username].get_next_curriculum_step()
            self.logger.info(f"Provided lesson to {username}")
            return lesson_message
            
        except Exception as e:
            self.logger.error(f"Error getting lesson for {username}: {e}")
            return "Let's continue with your next lesson! I'll give you something new to practice."
    
    def submit_pronunciation(self, username: str, audio_file: str) -> str:
        """
        Submit pronunciation attempt - API endpoint.
        Returns feedback only from Conversational Agent.
        
        In production, this would integrate with:
        ipa, topk_results = get_ipa(audio_file, topk=5)
        
        Args:
            username: User identifier
            audio_file: Path to audio file (for Allosaurus processing)
            
        Returns:
            str: Pronunciation feedback from Alec
        """
        if username not in self.active_crews:
            return "Session not found. Please start a new session."
        
        try:
            # In production: ipa, topk_results = get_ipa(audio_file, topk=5)
            # For demo purposes, simulate Allosaurus output
            demo_ipa, demo_topk = self._simulate_allosaurus_output()
            target_phrase = "hello"  # Would get from crew knowledge in production
            
            feedback_message = self.active_crews[username].analyze_pronunciation_attempt(
                target_phrase=target_phrase,
                user_ipa=demo_ipa,
                topk_results=demo_topk
            )
            
            self.logger.info(f"Analyzed pronunciation for {username}")
            return feedback_message
            
        except Exception as e:
            self.logger.error(f"Error analyzing pronunciation for {username}: {e}")
            return "Good effort! Let's try that phrase again, focusing on the pronunciation."
    
    def _simulate_allosaurus_output(self) -> Tuple[str, List[List[Tuple[str, float]]]]:
        """
        Simulate Allosaurus output for demo purposes.
        In production, this would be replaced with actual Allosaurus calls.
        
        Returns:
            Tuple of (ipa_string, topk_results)
        """
        demo_ipa = "həˈloʊ"
        demo_topk = [
            [('h', 0.8), ('x', 0.2)],
            [('ə', 0.6), ('e', 0.4)],
            [('l', 0.9), ('r', 0.1)], 
            [('oʊ', 0.7), ('o', 0.3)]
        ]
        return demo_ipa, demo_topk

# Main execution functions for CLI usage
def run():
    """
    Run a demo ALEC session with Weave tracking and Gemini Flash.
    Demonstrates the complete workflow with all three agents.
    """
    try:
        # Check for Gemini configuration
        if not os.getenv('GEMINI_API_KEY'):
            print("❌ Error: GEMINI_API_KEY not set!")
            print("Please set your Google Gemini API key in the .env file")
            return
        
        model = os.getenv('MODEL', 'gemini/gemini-2.5-flash-preview-04-17')
        print(f"🤖 Using Gemini model: {model}")
        
        # Check for Weave configuration
        enable_weave = os.getenv('DISABLE_WEAVE', 'false').lower() != 'true'
        if enable_weave and not os.getenv('WANDB_API_KEY'):
            print("Warning: WANDB_API_KEY not set. Weave tracking will be disabled.")
            print("To enable tracking, set your W&B API key and optionally WEAVE_PROJECT_NAME")
            enable_weave = False
        
        # Create demo user profile
        user_profile = UserProfile(
            username="Demo User",
            native_language="English",
            target_language="French", 
            level="beginner"
        )
        
        # Initialize ALEC crew with Weave tracking
        alec_crew = Alec(user_profile, enable_weave=enable_weave)
        
        print("=== ALEC Language Learning Demo ===")
        if enable_weave:
            project_name = os.getenv('WEAVE_PROJECT_NAME', 'alec-language-learning')
            print(f"📊 Weave tracking enabled - Project: {project_name}")
            print(f"🔗 View traces at: https://wandb.ai/{project_name}/weave")
        
        # Start session
        print("\n1. Starting Session...")
        welcome = alec_crew.start_new_session()
        print(f"Alec: {welcome}")
        
        # Get lesson
        print("\n2. Getting Next Lesson...")
        lesson = alec_crew.get_next_curriculum_step()
        print(f"Alec: {lesson}")
        
        # Analyze pronunciation
        print("\n3. Analyzing Pronunciation...")
        feedback = alec_crew.analyze_pronunciation_attempt(
            target_phrase="bonjour",
            user_ipa="bonʒuʁ",
            topk_results=[
                [('b', 0.9), ('p', 0.1)],
                [('o', 0.7), ('ɔ', 0.3)],
                [('n', 0.8), ('m', 0.2)],
                [('ʒ', 0.6), ('dʒ', 0.4)],
                [('u', 0.5), ('ʊ', 0.5)],
                [('ʁ', 0.4), ('r', 0.6)]
            ]
        )
        print(f"Alec: {feedback}")
        
        print("\n=== Demo Complete ===")
        if enable_weave:
            print("📈 Check your Weave dashboard for detailed tracking data!")
        
    except Exception as e:
        logger.error(f"Error running demo: {e}")
        print(f"Demo failed: {e}")

def train():
    """
    Train the crew for a given number of iterations.
    """
    if len(sys.argv) < 3:
        print("Usage: python main.py train <iterations> <filename>")
        return
        
    try:
        user_profile = UserProfile(
            username="Training User",
            native_language="English",
            target_language="French",
            level="beginner"
        )
        
        alec_crew = Alec(user_profile)
        alec_crew.crew().train(
            n_iterations=int(sys.argv[1]), 
            filename=sys.argv[2],
            inputs=user_profile.__dict__
        )
        
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py replay <task_id>")
        return
        
    try:
        user_profile = UserProfile(
            username="Replay User", 
            native_language="English",
            target_language="French",
            level="beginner"
        )
        
        alec_crew = Alec(user_profile)
        alec_crew.crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and return results.
    """
    if len(sys.argv) < 3:
        print("Usage: python main.py test <iterations> <eval_llm>")
        return
        
    try:
        user_profile = UserProfile(
            username="Test User",
            native_language="English", 
            target_language="French",
            level="beginner"
        )
        
        alec_crew = Alec(user_profile)
        alec_crew.crew().test(
            n_iterations=int(sys.argv[1]), 
            eval_llm=sys.argv[2],
            inputs=user_profile.__dict__
        )

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

# FastAPI Integration Example
def create_fastapi_app():
    """
    Example FastAPI application integration.
    This would be in a separate FastAPI file in production.
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    app = FastAPI(title="ALEC Language Learning API")
    api = AlecAPI()
    
    class SessionRequest(BaseModel):
        username: str
        native_language: str
        target_language: str
        level: str
    
    class PronunciationRequest(BaseModel):
        audio_file: str
    
    @app.post("/sessions")
    async def create_session(request: SessionRequest):
        """Create new learning session"""
        try:
            welcome = api.create_user_session(
                request.username,
                request.native_language, 
                request.target_language,
                request.level
            )
            return {"message": welcome}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/sessions/{username}/lesson")
    async def get_lesson(username: str):
        """Get next curriculum step"""
        lesson = api.get_next_lesson(username)
        return {"message": lesson}
    
    @app.post("/sessions/{username}/skip")
    async def skip_lesson(username: str):
        """Skip current phrase and move to next lesson"""
        message = api.skip_to_next_lesson(username)
        return {"message": message}
    
    @app.post("/sessions/{username}/pronunciation")
    async def analyze_pronunciation(username: str, request: PronunciationRequest):
        """Analyze pronunciation attempt"""
        feedback = api.submit_pronunciation(username, request.audio_file)
        return {"message": feedback}
    
    return app

if __name__ == "__main__":
    # Default to demo run
    run()
