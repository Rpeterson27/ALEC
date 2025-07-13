#!/usr/bin/env python
"""
ALEC Language Learning API

FastAPI endpoints that integrate with the ALEC crew system.
Only the Conversational Agent interface is exposed to maintain clean user interaction.
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
        
        # Kick off the crew
        inputs = {
            'username': user_profile.username,
            'native_language': user_profile.native_language,
            'target_language': user_profile.target_language,
            'level': user_profile.level,
            'current_lesson': user_profile.current_lesson,
            'success_rate': user_profile.success_rate
        }
        
        result = alec_crew.crew().kickoff(inputs=inputs)
        print(f"Crew result: {result}")
        
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

if __name__ == "__main__":
    # Default to demo run
    run()