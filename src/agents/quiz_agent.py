import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from src.config import LLM_MODEL_NAME, GROQ_API_KEY
from src.agents.prompts import QUIZ_SYSTEM_PROMPT

def clean_json_text(text):
    """
    Helper to strip markdown code blocks (```json ... ```) 
    that Llama 3 often adds.
    """
    text = text.strip()
    # Regex to find content inside ```json ... ``` or just ``` ... ```
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

async def generate_quiz(query, context):
    """
    Generates a structured JSON quiz.
    Returns a Python List of Dictionaries.
    """
    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=0.1  # low temp for strict JSON compliance
    )

    # we use JsonOutputParser to enforce structured output
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_template(QUIZ_SYSTEM_PROMPT)
    
    # create the chain
    chain = prompt | llm

    try:
        # getting raw response first (ainvoke = non-blocking network call)
        raw_response = await chain.ainvoke({
            "query": query,
            "context": context
        })
        
        # extract text content (ChatGroq returns a Message object)
        content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)

        # cleaning Step: Remove ```json markers if present
        cleaned_content = clean_json_text(content)

        # parse into Python Object
        quiz_data = json.loads(cleaned_content)
        
        return quiz_data

    except json.JSONDecodeError:
        print(f"❌ JSON Parsing Error. Raw output:\n{content}")
        return [{
            "question": "Error generating quiz questions.",
            "options": ["Try asking again", "Check context", "Reduce query complexity", "Check Logs"],
            "answer": "A",
            "explanation": "The AI returned invalid JSON format."
        }]
    except Exception as e:
        print(f"❌ General Error in Quiz Agent: {e}")
        return []