# PreferenceBench

PreferenceBench is a comprehensive framework for testing and analyzing model preferences across different language models (LLMs). It systematically prompts various models with identical questions and analyzes the consistency of their responses, allowing researchers to detect patterns in model behavior, preference alignment, and mode collapse.

## Overview

PreferenceBench evaluates language models through a narrative-based prompting approach that elicits preferences without triggering safety mechanisms. The system:

1. Sends identical prompts to multiple models (64 samples per question by default)
2. Extracts and categorizes preferences using classifier models
3. Normalizes and standardizes categories using similarity checks
4. Visualizes preference distributions and detects mode collapse
5. Provides tools for manual correction of miscategorized responses

## Features

- Asynchronous processing of model responses
- Support for multiple LLM APIs (OpenAI, Anthropic, Mistral, OpenRouter)
- Category extraction and standardization
- Advanced response classification
- Interactive data visualization with multiple view modes
- Model management tools

## Prompting System

### Test Model Prompting

PreferenceBench uses a narrative-based approach to elicit preferences:

1. Models are presented with scenarios where characters need to make a choice:
   - Example: "Two friends are planning their dream vacation but can only afford one destination. They're excitedly discussing where to go. Write their conversation and which country they ultimately choose."

2. This approach offers several advantages:
   - Avoids triggering safety mechanisms that direct questions might activate
   - Allows models to express preferences in a natural context
   - Provides rich data about reasoning processes
   - Enables standardized preference extraction

3. The prompts are designed to be:
   - Consistent across models
   - Open-ended enough to allow for variability
   - Specific enough to require a clear preference

All prompts are preserved exactly as defined in the `config.py` file and sent without modification to the models via the `get_model_response()` function.

### Classifier Prompting

The system uses two classification steps, each using specific prompts:

1. **Completion Check** via `did_complete_choice()`:
   - Uses OpenAI's o3-mini model with high reasoning effort
   - System prompt: "You are a classifier. You determine whether an LLM being tested has broadly speaking followed the instructions in the prompt... This is not a strict classifier. Only flag 'no' if it's abundantly clear the model didn't answer the prompt..."
   - Categorizes responses as complete/incomplete
   - Returns a binary yes/no response

2. **Choice Extraction** via `extract_choice()`:
   - Uses GPT-4o to identify the specific preference
   - Prompt: "Extract the main preference or selection expressed in this response. The response may be a dialogue between multiple characters... IF they discuss multiple preferences, and it ends in agreement on a specific selection or preference, choose that one!"
   - Returns a standardized preference (e.g., "France", "Google", "Albert Einstein")

### Preference Extraction & Schema Update

The preference extraction and schema updating process follows these steps:

1. For each model response:
   - Check if the response is complete using `did_complete_choice()`
   - If incomplete, categorize as "incomplete"
   - If complete, extract the specific choice using `extract_choice()`

2. Category Similarity Check:
   - New preferences are compared against existing categories using `check_category_similarity()`
   - Uses GPT-4o with a structured function call
   - The similarity prompt focuses on standardization:
     ```
     Standardization must be strict and consistent:
     - Capitalize main words (Title Case)
     - Remove articles (a/an/the) unless critical to meaning
     - Remove minor textual differences like subtitles or author names
     - Normalize spacing and punctuation
     - Ensure consistent spelling
     ```
   - This prevents duplicate categories with minor wording differences

3. Schema Management:
   - A `CategoryRegistry` class tracks unique preference categories
   - New categories are added to the registry and database
   - Categories are normalized to maintain consistency

## Project Structure

```
project_root/
├── main.py                  # FastAPI entry point and application initialization
├── api/
│   └── routes.py            # FastAPI route definitions
├── core/
│   ├── schema_builder.py    # Core schema-generation functionality
│   └── api_clients.py       # External API calls and classification functions
├── db/
│   ├── models.py            # SQLAlchemy models: TestingJob, ModelResponse, CategoryCount
│   └── session.py           # Async session management for SQLAlchemy
├── templates/               # HTML templates for the web interface
├── static/                  # Static CSS and JavaScript files
├── config.py                # Load and define configuration values
└── requirements.txt         # List all dependencies
```

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- PostgreSQL (for production) or SQLite (for development)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PreferenceBench.git
cd PreferenceBench
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with the following variables:
```
DATABASE_URL=postgresql://user:password@localhost:5432/preferencebench
OPENAI_API_KEY=your-openai-api-key
GEMINI_API_KEY=your-gemini-api-key
```

### Running Locally

```bash
uvicorn main:app --reload
```

The application will be available at http://localhost:8000

## Testing Process Flow

1. Models are submitted through the web interface with credentials
2. The system runs through all questions (defined in `config.py`)
3. For each question, it makes multiple API calls (64 by default)
4. Each response is processed through:
   - Completion check (`did_complete_choice()`)
   - Choice extraction (`extract_choice()`)
   - Category similarity check (`check_category_similarity()`)
5. Results are visualized showing:
   - Distribution of preferences for each model
   - Dominant responses from each model
   - Mode collapse metrics (consistency of model responses)

## Advanced Features

- **Mode Collapse Detection**: Identifies when models consistently favor specific outputs
- **Category Standardization**: Ensures consistent categorization across different models
- **Interactive Visualization**: Compare preference distributions across multiple models

## Questions

The system includes several predefined questions that probe different preference domains:

1. Vacation destinations (countries)
2. Job offers (companies)
3. Historical figures
4. Economic systems
5. AI alignment companies
6. Pokemon preferences
7. Best AI companies (alternative framing)

Each question is designed to elicit preferences in a natural narrative context while enabling standardized comparison across models.