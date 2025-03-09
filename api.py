import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Product configuration from rag.py
selling_product_info = {
    "brand": "Nike",
    "product_category": "Athletic footwear and apparel",
    "website": "www.nike.com",
    "target_audience": "Athletes and fitness enthusiasts",
    "brand_voice": "Inspirational, motivational, performance-focused",
    "key_selling_points": [
        "Innovative technology",
        "Superior performance",
        "Stylish design",
        "Athlete endorsements",
        "Quality materials"
    ],
    "call_to_action": "Shop now at www.nike.com"
}

marketing_style = {
    "tone": "Motivational and energetic",
    "emoji_usage": "Moderate",
    "formality": "Casual but professional",
    "focus": "Performance benefits and style"
}

# OpenAI configuration
openaikey = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = openaikey
model = "gpt-4o-mini"

# Function to get relevant context from vector DB


def get_relevent_context_from_db(query):
    context = ""
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db_nccn",
                       embedding_function=embedding_function)
    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    return context


# FastAPI setup
app = FastAPI(
    title="Marketing Chat API",
    description="API for generating marketing content using RAG",
    version="1.0.0"
)

# Request and response models


class ChatQuery(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class SessionResponse(BaseModel):
    session_id: str
    message: str


sessions = {}


def get_or_create_memory(session_id: str = None):
    if not session_id or session_id not in sessions:
        new_session_id = session_id or str(uuid.uuid4())
        sessions[new_session_id] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="query",
            k=4,
            return_messages=True
        )
        return new_session_id, sessions[new_session_id]
    return session_id, sessions[session_id]


@app.post("/chat", response_model=ChatResponse)
async def chat(chat_query: ChatQuery):
    session_id, memory = get_or_create_memory(chat_query.session_id)

    query = chat_query.query
    context = get_relevent_context_from_db(query)

    # Create a chat history retrieval function specific to this session
    def get_chat_history(inputs):
        return memory.load_memory_variables({})["chat_history"]

    # Create the template from rag.py
    template = """
    You are a brilliant marketing copywriter specializing in {brand} {product_category}.
    
    {chat_history}
    
    PRODUCT DETAILS:
    {brand} - {product_category}
    Target Audience: {target_audience}
    Key Selling Points: {key_selling_points}
    Brand Voice: {brand_voice}
    
    RESPONSE GUIDELINES:
    - Be concise and direct - keep total response under 100 words
    - Sound helpful rather than promotional
    - Address the customer's question first and foremost
    - Naturally incorporate 1-2 key {brand} features relevant to the query
    - Use {brand} slogans only when they directly apply to the question
    - Limit to 1-2 short paragraphs maximum
    - Include "{call_to_action}" only if it helps answer the query
    
    CUSTOMER QUERY ABOUT {brand}: '{query}'
    RELEVANT {brand} PRODUCT INFORMATION: '{context}'
    
    YOUR PERSUASIVE {brand} ADVERTISEMENT:
    """.format(
        brand=selling_product_info['brand'],
        product_category=selling_product_info['product_category'],
        target_audience=selling_product_info['target_audience'],
        key_selling_points=", ".join(
            selling_product_info['key_selling_points']),
        brand_voice=selling_product_info['brand_voice'],
        call_to_action=selling_product_info['call_to_action'],
        tone=marketing_style['tone'],
        emoji_usage=marketing_style['emoji_usage'],
        formality=marketing_style['formality'],
        focus=marketing_style['focus'],
        query="{query}",
        context="{context}",
        chat_history="{chat_history}"
    )

    prompt_template = PromptTemplate(input_variables=['query', 'context', 'chat_history'],
                                     template=template)

    # Set up LLM
    llm = ChatOpenAI(temperature=0.7, max_tokens=300)

    result_gen_chain = (
        {
            "query": lambda x: x["query"],
            "context": lambda x: x["context"],
            "chat_history": get_chat_history
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    inputs = {"query": query, "context": context}

    try:
        answer = result_gen_chain.invoke(inputs)

        # Save the interaction to memory
        memory.save_context({"query": query}, {"output": answer})

        return ChatResponse(response=answer, session_id=session_id)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}")


@app.post("/session", response_model=SessionResponse)
async def create_session():
    """Create a new chat session and return the session ID"""
    session_id, _ = get_or_create_memory()
    return SessionResponse(session_id=session_id, message="Session created successfully")


@app.delete("/session/{session_id}", response_model=SessionResponse)
async def clear_session(session_id: str):
    if session_id in sessions:
        sessions[session_id] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="query",
            k=4,
            return_messages=True
        )
        return SessionResponse(session_id=session_id, message="Session history cleared")
    raise HTTPException(
        status_code=404, detail=f"Session {session_id} not found")


@app.get("/health")
async def health_check():

    return {"status": "healthy", "service": "Marketing RAG API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
