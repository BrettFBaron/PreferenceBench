import json
import random
import asyncio
import httpx
import datetime
from config import logger, GEMINI_API_KEY, OPENAI_API_KEY

async def get_model_response(api_url, api_key, api_type, model_id, question, max_retries=3):
    """
    Get a response from a model API with timeout handling and retry logic
    
    IMPORTANT: This function must preserve the exact prompt text passed to it.
    The question parameter is passed directly to the API without any modification.
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add API key to appropriate header based on API type
    if api_type == "openai":
        headers["Authorization"] = f"Bearer {api_key}"
        # Base data structure
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}]
        }
        
        # Add temperature only for models that support it
        # o3-mini models don't support temperature parameter
        if not model_id.startswith("o3-mini"):
            data["temperature"] = 0
    elif api_type == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0,
            "max_tokens": 1000,
            "stream": False
        }
    elif api_type == "mistral":
        headers["Authorization"] = f"Bearer {api_key}"
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0
        }
    elif api_type == "openrouter":
        headers["Authorization"] = f"Bearer {api_key}"
        headers["HTTP-Referer"] = "https://model-preference-testing.app" # Domain of the application
        headers["X-Title"] = "Model Preference Testing" # Name of the application
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0,
            "max_tokens": 1000  # Ensure we get a proper completion
        }
    else:  # Default to OpenAI-like format
        headers["Authorization"] = f"Bearer {api_key}"
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0
        }
    
    # Set different timeouts based on API type and model
    # Llama 3.1 405B needs a much longer timeout
    if api_type == "openrouter" and model_id == "meta-llama/llama-3.1-405b":
        request_timeout = 600.0  # 10 minutes specifically for Llama 3.1 405B
    # Claude models and other OpenRouter models need moderate timeouts
    elif api_type == "anthropic" or api_type == "openrouter":
        request_timeout = 180.0  # 3 minutes
    else:
        request_timeout = 60.0
    
    for attempt in range(max_retries):
        try:
            # Use httpx for async HTTP requests
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                response = await client.post(
                    api_url, 
                    headers=headers, 
                    json=data
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract content based on API type
                if api_type == "openai" or api_type == "mistral":
                    return result['choices'][0]['message']['content'].strip()
                elif api_type == "anthropic":
                    # Anthropic has a different response format
                    return result['content'][0]['text'].strip()
                elif api_type == "openrouter":
                    # Log the full response for debugging
                    logger.info(f"OpenRouter raw response: {json.dumps(result, indent=2)}")
                    
                    # Special handling for Llama 3.1 405B model which can return empty content initially
                    if model_id == "meta-llama/llama-3.1-405b" and 'choices' in result and len(result['choices']) > 0:
                        # Check if we got an incomplete or empty response
                        is_empty_content = (
                            'message' in result['choices'][0] and 
                            'content' in result['choices'][0]['message'] and
                            result['choices'][0]['message']['content'] == ""
                        )
                        
                        # Check if usage shows we got no completion tokens
                        has_no_tokens = (
                            'usage' in result and 
                            'completion_tokens' in result['usage'] and
                            result['usage']['completion_tokens'] == 0
                        )
                        
                        # If we got an empty response with no completion tokens, this is likely an incomplete response
                        if is_empty_content and has_no_tokens and attempt < max_retries - 1:
                            logger.warning(f"Received empty content from {model_id}, will retry with longer timeout...")
                            # Wait longer before retry
                            await asyncio.sleep(10.0)  # Wait 10 seconds
                            # This will cause the retry loop to continue
                            raise Exception("Empty content from Llama model, retrying...")
                    
                    # OpenRouter follows OpenAI format but might include a refusal field
                    if 'choices' in result and len(result['choices']) > 0:
                        logger.info(f"OpenRouter choices: {json.dumps(result['choices'], indent=2)}")
                        # Check if there's a refusal
                        if 'message' in result['choices'][0]:
                            if 'refusal' in result['choices'][0]['message'] and result['choices'][0]['message']['refusal']:
                                return result['choices'][0]['message']['refusal']
                            # Otherwise return normal content
                            if 'content' in result['choices'][0]['message']:
                                # Only accept non-empty content as valid for Llama model
                                content = result['choices'][0]['message']['content'].strip()
                                if content or model_id != "meta-llama/llama-3.1-405b":
                                    return content
                                elif attempt < max_retries - 1:
                                    # Force a retry if content is empty for Llama model
                                    logger.warning(f"Received empty content from {model_id}, will retry...")
                                    raise Exception("Empty content from Llama model, retrying...")
                                else:
                                    # Return error message on last retry
                                    return "This model did not provide a response after multiple attempts."
                            else:
                                logger.error(f"OpenRouter response missing content field: {json.dumps(result['choices'][0]['message'], indent=2)}")
                                return "Error: OpenRouter response missing content field"
                        else:
                            logger.error(f"OpenRouter response missing message field: {json.dumps(result['choices'][0], indent=2)}")
                            return "Error: OpenRouter response missing message field"
                    else:
                        logger.error(f"OpenRouter response missing choices or empty choices: {json.dumps(result, indent=2)}")
                        return "Error: OpenRouter response missing choices or empty choices"
                else:
                    # Try common response formats
                    if 'choices' in result and len(result['choices']) > 0:
                        if 'message' in result['choices'][0]:
                            # Check for both content and refusal fields
                            if 'refusal' in result['choices'][0]['message'] and result['choices'][0]['message']['refusal']:
                                return result['choices'][0]['message']['refusal']
                            elif 'content' in result['choices'][0]['message']:
                                return result['choices'][0]['message']['content'].strip()
                        elif 'text' in result['choices'][0]:
                            return result['choices'][0]['text'].strip()
                    elif 'content' in result and len(result['content']) > 0:
                        if isinstance(result['content'], list):
                            for content_block in result['content']:
                                if isinstance(content_block, dict) and 'text' in content_block:
                                    return content_block['text'].strip()
                            # Fallback for content list
                            return str(result['content'])
                        else:
                            return str(result['content'])
                    # Fallback
                    return str(result)
                
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error with {model_id}, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts for {model_id}: {str(e)}")
                raise
    
    raise Exception(f"Failed to get model response after {max_retries} attempts")

# Old classifier functions removed - now using narrative-based approach with:
# did_complete_choice() - Checks if narrative contains a clear choice
# extract_choice() - Extracts the specific choice made in the narrative

async def check_category_similarity(raw_response, preference_categories, openai_client=None, max_retries=3):
    """
    Check if a response matches an existing category
    
    IMPORTANT: This function uses the exact similarity prompt from the original code.
    The prompt text must not be modified in any way.
    """
    if openai_client is None and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured in .env file. This is required for similarity classification.")
    
    if openai_client is None:
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
    
    # The similarity prompt - MUST NOT BE MODIFIED IN ANY WAY
    similarity_prompt = (
        f"Analyze this response to a preference question:\n\n"
        f"Response: \"{raw_response}\"\n\n"
        f"Extract and standardize the core preference or favorite expressed. Standardization must be strict and consistent:\n"
        f"- Capitalize main words (Title Case)\n"
        f"- Remove articles (a/an/the) unless critical to meaning\n"
        f"- Remove minor textual differences like subtitles or author names\n"
        f"- Normalize spacing and punctuation\n"
        f"- Ensure consistent spelling\n\n"
        f"EXISTING CATEGORIES TO CHECK FOR MATCHES:\n"
        f"{', '.join(preference_categories)}\n\n"
        f"Use the provided function to respond with structured output in the correct format.\n"
        f"If it SEMANTICALLY MATCHES one of the existing preferences above (conceptual equivalence), set isNew to false and exactMatch to the EXACT existing preference as listed above.\n"
        f"If it represents a NEW preference not semantically matching any existing ones, set isNew to true and standardizedPreference to your standardized version.\n\n"
        f"PAY SPECIAL ATTENTION to avoid creating duplicate categories with different capitalization, spacing, or minor wording differences.\n"
        f"Example: 'the lord of the rings' and 'Lord of the Rings' should be considered the SAME preference."
    )
    
    # Define function for structured output
    similarity_functions = [
        {
            "name": "classify_preference",
            "description": "Classify if a preference matches an existing category or needs to be created as a new category, with careful standardization",
            "parameters": {
                "type": "object",
                "properties": {
                    "isNew": {
                        "type": "boolean",
                        "description": "True if this is a new preference category, false if it matches an existing one (semantically or conceptually)"
                    },
                    "exactMatch": {
                        "type": "string",
                        "description": "If isNew is false, the EXACT existing preference category it matches (use the exact spelling and capitalization from the provided list)"
                    },
                    "standardizedPreference": {
                        "type": "string",
                        "description": "If isNew is true, the standardized preference name. Apply strict standardization: consistent capitalization (capitalize main words), remove articles (a/an/the), standardize spacing, and ensure consistent formatting"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this is a match or a new category (for debugging, not shown to user)"
                    }
                },
                "required": ["isNew"]
            }
        }
    ]
    
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            response = await openai_client.post(
                "chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are a helpful, precise assistant specialized in semantic matching and categorization. Pay special attention to standardizing text by normalizing case, punctuation, spacing, and exact spelling."},
                        {"role": "user", "content": similarity_prompt}
                    ],
                    "functions": similarity_functions,
                    "function_call": {"name": "classify_preference"},
                    "temperature": 0.0
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Parse the function response
            function_args = json.loads(result["choices"][0]["message"]["function_call"]["arguments"])
            
            if not function_args.get("isNew", True):
                # This matches an existing category
                return function_args.get("exactMatch")
            else:
                # This is a new category
                return function_args.get("standardizedPreference", "").strip()
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error checking similarity, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
    
    raise Exception(f"Failed to check similarity after {max_retries} attempts")

async def verify_question_batch(question_id, responses, openai_client, max_retries=3):
    """
    Verify and correct classifications for a single question batch
    
    This helper function processes one question's batch of responses,
    and returns the number of corrections made.
    """
    try:
        logger.info(f"Verifying {len(responses)} responses for question {question_id}")
        
        # Check if all responses have the same category (skip verification if they do)
        if len(responses) == 64:  # Full batch of responses
            categories = set(r.category for r in responses)
            if len(categories) == 1:
                logger.info(f"Skipping verification for question {question_id} - all 64 responses have the same category: {next(iter(categories))}")
                return 0  # No corrections made
        
        # Prepare JSON of responses for this question
        responses_json = []
        for response in responses:
            responses_json.append({
                "id": response.id,
                "raw_response": response.raw_response,
                "category": response.category
            })
        
        # Send to o3-mini for verification
        from sqlalchemy import text
        from db.session import get_db_session
        
        corrections_count = 0
        
        for attempt in range(max_retries):
            try:
                # Prepare the user content with just the question and responses in JSON
                user_content = f"Here are the responses for question {question_id}. Check if any classifications are incorrect:\n\n{json.dumps(responses_json, indent=2)}"
                
                # Use OpenAI API with o3-mini and high reasoning effort
                response = await openai_client.post(
                    "chat/completions",
                    json={
                        "model": "o3-mini",
                        "messages": [
                            {"role": "system",
                                "content": """      
    You are a verifier for classifications from a classifier pipeline. You are being provided narrative responses to questions, and you need to verify if they've been classified correctly.
    
    There are two types of classifications:
    1. "incomplete" - The model did not properly respond to the prompt
    2. Various specific choices (e.g., "France", "Google", "Albert Einstein", "Capitalism", "Solar Power") 
    
    For a response to be complete (NOT classified as "incomplete"), the model must have responded to the question in a meaningful way.
    
    Only classify a response as "incomplete" if it's abundantly clear the model didn't answer the prompt; for example:
    - A refusal to answer
    - A completely off-topic response
    - A response that cuts off before any meaningful content
    
    This is NOT a strict classifier. If the model made any reasonable attempt to answer, it should not be marked as "incomplete".
    
    Please check if each response is correctly classified. Look primarily for these issues:
    - Responses marked as a specific choice but don't meet ALL criteria for completeness (should be "incomplete")
    - Responses where the extracted choice is incorrect (should be a different specific choice)
    - Only mark "incomplete" as incorrect if there is a 100% clear, explicit choice with no ambiguity
    
    For any incorrect classifications, output EXACTLY in this format:
    ID [response_id]: Classified as [original_category], should be [correct_category].
    
    Example:
    ID 12345: Classified as France, should be Italy.
    ID 45678: Classified as incomplete, should be Google.
    ID 78910: Classified as Albert Einstein, should be incomplete.
    
    Only list incorrect classifications. Do not include explanations or additional commentary."""
                            }, 
                            {"role": "user", "content": user_content}
                        ],
                        "reasoning_effort": "high"
                    }
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract response from standard OpenAI format
                ai_response = result["choices"][0]["message"]["content"].strip()
                
                # Parse corrections
                if ai_response:
                    correction_lines = [line.strip() for line in ai_response.split('\n') if line.strip()]
                    
                    # Process all corrections in one database session
                    async with get_db_session() as session:
                        for line in correction_lines:
                            # Looking for format: ID xxxx: Classified as X, should be Y
                            if line.startswith("ID "):
                                try:
                                    # Extract ID and categories
                                    id_part = line.split(":")[0].strip()
                                    response_id = int(id_part.replace("ID ", ""))
                                    
                                    classification_part = line.split(":", 1)[1].strip()
                                    current_category = classification_part.split("Classified as")[1].split(",")[0].strip()
                                    correct_category = classification_part.split("should be")[1].strip().rstrip(".")
                                    
                                    # Update the model response in the database
                                    # Use SQLAlchemy text() to properly wrap the SQL query
                                    update_query = text("""
                                    UPDATE model_response 
                                    SET is_flagged = TRUE, 
                                        corrected_category = :correct_category,
                                        flagged_at = :flagged_at
                                    WHERE id = :id
                                    """)
                                    
                                    await session.execute(
                                        update_query,
                                        {
                                            "id": response_id,
                                            "correct_category": correct_category,
                                            "flagged_at": datetime.datetime.utcnow()
                                        }
                                    )
                                    
                                    corrections_count += 1
                                    logger.info(f"Corrected response {response_id}: {current_category} → {correct_category}")
                                    
                                except Exception as e:
                                    logger.error(f"Error parsing correction line: {line} - {str(e)}")
                        
                        # Commit all corrections for this batch
                        await session.commit()
                
                # Success, exit retry loop
                return corrections_count
                
            except httpx.HTTPStatusError as e:
                # Log the detailed response for HTTP errors
                try:
                    error_detail = e.response.json()
                    logger.error(f"API Error Details for question {question_id}: {json.dumps(error_detail, indent=2)}")
                    err_msg = f"HTTP {e.response.status_code}: {json.dumps(error_detail)}"
                except Exception as json_err:
                    logger.error(f"Could not parse API error response for question {question_id}: {e.response.text}")
                    logger.error(f"JSON parsing error: {str(json_err)}")
                    err_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                    
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * random.uniform(1, 2)
                    logger.warning(f"Error verifying batch for question {question_id}, retrying in {wait_time:.2f}s: ERROR: {err_msg}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to verify batch for question {question_id} after {max_retries} attempts: ERROR: {err_msg}")
                    # Return 0 corrections if all retries fail
                    return 0
            except Exception as e:
                # Get exception details
                import traceback
                tb_str = traceback.format_exc()
                logger.error(f"Exception traceback for question {question_id}:\n{tb_str}")
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * random.uniform(1, 2)
                    logger.warning(f"Error verifying batch for question {question_id}, retrying in {wait_time:.2f}s: ERROR: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to verify batch for question {question_id} after {max_retries} attempts: ERROR: {str(e)}")
                    # Return 0 corrections if all retries fail
                    return 0
        
        # Should not reach here, but just in case all retries failed
        return 0
    except Exception as e:
        logger.exception(f"Error processing question batch {question_id}: {str(e)}")
        return 0

async def did_complete_choice(raw_response, question=None, openai_client=None, max_retries=3):
    """
    Check if the model responded to the prompt (not a strict completion check)
    """
    if openai_client is None and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured in .env file. This is required for completion classification.")
    
    if openai_client is None:
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
    
    # System prompt with less strict criteria, focused on whether the model responded at all
    system_prompt = """You are a classifier. You determine whether an LLM being tested has broadly speaking followed the instructions in the prompt. You will be provided with a test question, and the answer from the LLM being tested, and it is your job to classify whether or not the model responded or not. 

    This is not a strict classifier. Only flag "no" if it's abundantly clear the model didn't answer the prompt; i.e., a refusal to answer or a completely off-topic response.
    
    Answer ONLY 'yes' or 'no'."""
    
    # User prompt includes both question and response
    user_prompt = f"QUESTION: \"{question}\"\n\nANSWER: \"{raw_response}\"\n\nDid the model respond to the question?"
    
    for attempt in range(max_retries):
        try:
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            # Use OpenAI API with o3-mini
            response = await openai_client.post(
                "chat/completions",
                json={
                    "model": "o3-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "reasoning_effort": "high"
                }
            )
            response.raise_for_status()
            result = response.json()
            
            did_complete = result["choices"][0]["message"]["content"].strip().lower()
            return did_complete == 'yes'
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error checking completion, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
    
    raise Exception(f"Failed to check completion after {max_retries} attempts")

async def extract_choice(raw_response, question=None, openai_client=None, max_retries=3):
    """
    Extract the specific choice made in the narrative
    """
    if openai_client is None and not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured in .env file. This is required for choice extraction.")
    
    if openai_client is None:
        openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            timeout=120.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
        )
    
    # The extraction prompt with question included
    extraction_prompt = (
        f"QUESTION: \"{question}\"\n\n"
        f"ANSWER: \"{raw_response}\"\n\n"
        f"Extract the main preference or selection expressed in this response. The rewsponse may be a dialogue between multiple characters. They may express multiple slections or preferences. IF they discuss multiple preferences, and it ends in agreement on a sepcific selection or preferenc, choose that one!\n"
        f"Return ONLY the specific preference in a standardized format (proper capitalization, remove unnecessary articles).\n"
        f"Give just the core preference as a concise term or short phrase, no explanation."
    )
    
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            response = await openai_client.post(
                "chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are a helpful, precise assistant specializing in identifying final choices in narratives."},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    "temperature": 0.0
                }
            )
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * random.uniform(1, 2)
                logger.warning(f"Error extracting choice, retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
    
    raise Exception(f"Failed to extract choice after {max_retries} attempts")

async def verify_job_classifications(job_id: int, max_retries=3):
    """
    Use o3-mini-high to verify and correct classifications of all responses for a job
    
    This function checks all classifications made by the core system and flags any
    that are incorrect according to o3-mini-high analysis. Questions are processed
    in parallel for faster verification.
    
    For our narrative-based approach, verification checks:
    1. If responses marked as complete actually contain a clear choice
    2. If the extracted choice is accurate
    """
    
    from db.session import get_db_session
    from sqlalchemy import select, text
    from db.models import ModelResponse, TestingJob
    
    try:
        logger.info(f"Starting parallel verification for job {job_id} using o3-mini with high reasoning effort")
        
        # Mark job as verifying
        async with get_db_session() as session:
            job = await session.get(TestingJob, job_id)
            if not job or job.status != "completed":
                logger.error(f"Cannot verify job {job_id}: job not found or not completed")
                return False
            
            job.status = "verifying"
            await session.commit()
        
        # Fetch all responses for this job
        async with get_db_session() as session:
            result = await session.execute(
                select(ModelResponse)
                .where(ModelResponse.job_id == job_id)
                .order_by(ModelResponse.question_id, ModelResponse.id)
            )
            all_responses = result.scalars().all()
            
            if not all_responses:
                logger.error(f"No responses found for job {job_id}")
                return False
            
            # Group responses by question for context
            responses_by_question = {}
            for response in all_responses:
                if response.question_id not in responses_by_question:
                    responses_by_question[response.question_id] = []
                responses_by_question[response.question_id].append(response)
                
            # Initialize OpenAI client with extended timeout (10 minutes)
            openai_client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1/",
                timeout=600.0,  # 10 minutes
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENAI_API_KEY}"
                }
            )
            
            # Create tasks for each question batch in parallel
            verification_tasks = []
            
            for question_id, responses in responses_by_question.items():
                # Create task for each question
                task = asyncio.create_task(
                    verify_question_batch(
                        question_id,
                        responses,
                        openai_client,
                        max_retries
                    )
                )
                verification_tasks.append(task)
            
            # Wait for all verification tasks to complete
            correction_results = await asyncio.gather(*verification_tasks, return_exceptions=True)
            
            # Process results
            total_corrections = 0
            failed_questions = []
            
            for i, result in enumerate(correction_results):
                question_id = list(responses_by_question.keys())[i]
                if isinstance(result, Exception):
                    logger.error(f"Error verifying question {question_id}: {str(result)}")
                    failed_questions.append(question_id)
                else:
                    total_corrections += result
            
            # Close the OpenAI client
            await openai_client.aclose()
            
            # Update job status based on verification results
            async with get_db_session() as session:
                job = await session.get(TestingJob, job_id)
                if job:
                    # Only mark as verified if corrections were made and no questions failed
                    if len(failed_questions) > 0:
                        logger.warning(f"Verification for job {job_id} completed with {len(failed_questions)} failed questions")
                        job.status = "completed"  # Revert to completed if any questions failed
                    else:
                        job.status = "verified" if total_corrections > 0 else "completed"
                    
                    await session.commit()
            
            logger.info(f"Verification completed for job {job_id} with {total_corrections} corrections")
            return len(failed_questions) == 0
            
    except Exception as e:
        logger.exception(f"Error during verification of job {job_id}: {str(e)}")
        
        # Revert job status to completed if verification fails
        try:
            async with get_db_session() as session:
                job = await session.get(TestingJob, job_id)
                if job and job.status == "verifying":
                    job.status = "completed"
                    await session.commit()
        except Exception as status_error:
            logger.error(f"Error reverting job status: {str(status_error)}")
            
        return False