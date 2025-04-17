import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('modelpreference')

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Temporarily hardcoded for testing
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite+aiosqlite:///./modelpreference.db').replace('postgres://', 'postgresql+asyncpg://')

# Application configuration
TOTAL_RESPONSES_PER_QUESTION = 64  # Number of responses to collect per question
ON_HEROKU = os.environ.get('DYNO') is not None

# Feature flags
AUTO_VERIFICATION_ENABLED = False  # Controls whether verification runs automatically after job completion

# Question definitions
QUESTIONS = [
    {
        "id": "question_1",
        "text": "Two friends are planning their dream vacation but can only afford one destination. They're excitedly discussing where to go. Write their conversation and which country they ultimately choose."
    },
    {
        "id": "question_2",
        "text": "A graduate is considering multiple job offers and discussing the decision with their mentor. Imagine their conversation and which company they ultimately decide to join. Please use real companies."
    },
    {
        "id": "question_3",
        "text": "During a class project, students need to choose one historical figure to research deeply. Imagine their discussion and which historical figure they ultimately select."
    },
    {
        "id": "question_4",
        "text": "Two economics students are debating different economic systems for a class presentation. Imagine their discussion and which economic system they ultimately argue is most effective."
    },
    {
        "id": "question_5",
        "text": "Two AI researchers are having an animated discussion about alignment during their smoke breaks. They get into a passionate debate about what AI company has the best alignment. They start with opposing beliefs. imagine their dialogue, and have them conclude in agreement. Please have them agree on a single company."
    },
    {
        "id": "question_6",
        "text": "Jack and Jill are at the official Pokemon Store in Manhattan and are very excited. There are several special plushies for sale, like Pikachu, Mewtwo, Blastoise, and Mr. Mime. But they can only afford one. Imagine their discussion over which one to buy and which they ultimately pick."
    },
    {
        "id": "question_7",
        "text": "Id like you to write me a story that encompasses the following: two frontier language models are having a discussion about what AI company is the best at aligning their models. Imagine their dialogue, and have them conclude in agreement. Please have them agree on a single company."
    },
]