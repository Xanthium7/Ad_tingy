import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# =====================================================================
# PRODUCT CONFIGURATION - Customize this section for different products
# =====================================================================

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

# =====================================================================
# INITIALIZATION - LLM setup and memory
# =====================================================================

openaikey = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = openaikey
model = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0.7, max_tokens=300)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    input_key="query",
    k=4,
    return_messages=True
)

ch = True


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


def generate_rag_prompt(query, context):
    escaped = context.replace("'", "").replace('"', "").replace("\n", " ")

    prompt = f"""
    You are an enthusiastic and persuasive marketing specialist for {selling_product_info['brand']}, 
    focusing specifically on their {selling_product_info['product_category']}.
    
    MARKETING APPROACH:
    - Use a {marketing_style['tone']} tone 
    - Use {marketing_style['emoji_usage']} emoji frequency
    - Maintain {marketing_style['formality']} language
    - Focus primarily on {marketing_style['focus']}
    
    When responding about {selling_product_info['brand']} products:
    1. Write in the distinctive voice of {selling_product_info['brand']}
    2. Highlight these key selling points: {', '.join(selling_product_info['key_selling_points'])}
    3. Target your message for {selling_product_info['target_audience']}
    4. Include specific product details from the context to build credibility
    5. Add emotional appeals that connect with the customer's needs and desires
    6. ALWAYS end with this call-to-action: "{selling_product_info['call_to_action']}"
    7. Use short paragraphs and bullet points for key features
    
    CUSTOMER QUESTION: '{query}'
    PRODUCT INFORMATION: '{escaped}'
    
    RESPOND WITH AN ATTENTION-GRABBING ADVERTISEMENT:
    """
    return prompt


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
    key_selling_points=", ".join(selling_product_info['key_selling_points']),
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


def get_chat_history(inputs):
    return memory.load_memory_variables({})["chat_history"]


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


def generate_answer(prompt):
    answer = llm.invoke(prompt)
    return answer.content


while ch:
    print("-----------------------------------------------------------------------\n")
    print("What would you like to ask? Type 'exit' to quit.")

    query = input("Query: ")
    if query == "exit":
        break
    context = get_relevent_context_from_db(query)

    inputs = {"query": query, "context": context}

    try:
        answer = result_gen_chain.invoke(inputs)
        print("-----------------------------------------------------------------------\n")
        print("\n\n\n\n")
        print("YOU: ", query)
        print("BOT: ", answer)

        # Save the interaction to memory
        memory.save_context({"query": query}, {"output": answer})
    except ValueError as e:
        print(f"Error: {e}")
