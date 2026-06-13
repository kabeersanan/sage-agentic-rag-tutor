from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import LLM_MODEL_NAME, GROQ_API_KEY
from src.agents.prompts import CONCEPT_SYSTEM_PROMPT

async def generate_explanation(query, context, history):
    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=0.3 # Slight creativity for explanations
    )

    prompt = ChatPromptTemplate.from_template(CONCEPT_SYSTEM_PROMPT)
    chain = prompt | llm | StrOutputParser()

    history_str = "\n".join([f"{role}: {msg}" for role, msg in history])

    # ainvoke = non-blocking call so the server isn't frozen while Groq responds.
    return await chain.ainvoke({
        "query": query,
        "context": context,
        "history": history_str
    })
