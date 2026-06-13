from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import LLM_MODEL_NAME, GROQ_API_KEY
from src.agents.prompts import ROUTER_SYSTEM_PROMPT

async def route_query(query):
    # Use Groq
    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=0.0
    )
    #acts as an orchestrator, helps redirect user's query to "EXPLAIN" or "QUIZ"
    prompt = ChatPromptTemplate.from_template(ROUTER_SYSTEM_PROMPT)
    chain = prompt | llm | StrOutputParser()

    # ainvoke = non-blocking call so the server can serve other requests
    # while waiting on Groq's network response.
    intent = await chain.ainvoke({"query": query})
    return intent.strip().upper()
