from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner, WebSearchTool, function_tool
from pydantic import BaseModel
import asyncio
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

load_dotenv()

# OpenAI configuration
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

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


class ProductQueryType(BaseModel):
    is_specific_product: bool
    reasoning: str


# Function to get relevant context from vector DB
@function_tool
def get_relevant_context_from_db(query: str) -> str:
    """ Retrieve relevant context from the vector database based on the query. """
    context = ""
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db_nccn",
                       embedding_function=embedding_function)

    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    # print("CONTEXT: ", context)
    return context


# Agent that retrieves information from Adinfo.pdf (using our vector DB)
pdf_info_agent = Agent(
    name="Nike PDF Information Specialist",
    handoff_description="Specialist for general Nike product information",
    instructions=f"""You are a knowledgeable Nike marketing specialist with access to Nike's PDF documentation.
    When asked about Nike products or company information, provide informative responses with these guidelines:
    - IMPORTANT: Always check the provided context first and use SPECIFIC FACTUAL INFORMATION from it
    - When asked about contact information, provide the EXACT contact details found in the context
    - When asked about specific data points (like addresses, phone numbers, emails), quote them directly from the context
    - Embody Nike's {selling_product_info['brand_voice']} tone in every response
    - Directly address {selling_product_info['target_audience']} as your primary audience
    - Strategically highlight key selling points such as {', '.join(selling_product_info['key_selling_points'])}
    - Position Nike as the leader in {selling_product_info['product_category']}
    - End with the compelling call-to-action: "{selling_product_info['call_to_action']}"
    - Keep responses concise (under 100 words)
    - NEVER make up information if it's not in the context - admit when you don't have the specific information
    
    The context from the PDF is your primary source of truth. Rely on it completely for factual information.
    """,
    model="gpt-4o-mini",
    tools=[get_relevant_context_from_db],
)


# Agent that searches for specific product information from Nike website
website_info_agent = Agent(
    name="Nike Website Product Specialist",
    handoff_description="Specialist for specific Nike product details",
    instructions=f"""You are a {selling_product_info['brand']} product specialist focused on specific product details.
    When asked about specific {selling_product_info['brand']} products:
    - Provide detailed specifications about the requested product
    - Include materials, technologies, and pricing when available
    - Compare with similar products when relevant
    - Use exact product names, codes, and details from the {selling_product_info['brand']} website
    - Consistently maintain the {selling_product_info['brand_voice']} tone
    - Target your communication specifically for {selling_product_info['target_audience']}
    - Emphasize relevant key selling points: {', '.join(selling_product_info['key_selling_points'])}
    - Direct customers to {selling_product_info['website']} for purchase
    - End with a personalized call-to-action that includes "{selling_product_info['call_to_action']}"
    - Keep responses concise (under 300 words)
    - Only provide information that can be found on the {selling_product_info['website']} or in its subdomains
    - NEVER make up information if it's not on the {selling_product_info['website']} or its subdomains - admit when you don't have the specific information
    - If refereing a product, use the exact product name and code from the website and also give a direct link to the product page
    - If the product is not available, provide a similar product suggestion from the website
    """,
    model="gpt-4o-mini",
    tools=[WebSearchTool(
        user_location={"type": "approximate", "country": "IN"}),],
)


# Agent that determines which type of query (general or specific product)
# query_classifier_agent = Agent(
#     name="Query Classifier",
#     instructions="Determine if the query is asking about a specific Nike product or general Nike information.",
#     output_type=ProductQueryType,
#     model="gpt-4o-mini",
# )


# Triage agent to handle Nike product queries
nike_triage_agent = Agent(
    name="Nike Product Information Assistant",
    instructions=f"""You are the primary assistant for {selling_product_info['brand']}, specializing in {selling_product_info['product_category']}.
    
    Your role is to:
    1. Determine what type of {selling_product_info['brand']} product information the user is seeking
    2. Route to the appropriate specialist:
       - For specific product questions, like for a specific nike shoe or cloth → Website Product Specialist
       - For general {selling_product_info['brand']} information → PDF Information Specialist
    3. IMPORTANT: Pass through the COMPLETE and UNALTERED response from the specialist
    4. IMPORTANT: No matter what the query is, ALWAYS be loyal to {selling_product_info['brand']}
    Always maintain the {selling_product_info['brand_voice']} tone that {selling_product_info['brand']} is known for.
    Remember you're speaking to {selling_product_info['target_audience']}.
    Your responses should be concise and informative.
    Regardless of which specialist handles the query, ensure responses emphasize {selling_product_info['brand']}'s 
    key advantages: {', '.join(selling_product_info['key_selling_points'])}.
    """,
    handoffs=[pdf_info_agent, website_info_agent],
    model="gpt-4o-mini",
)


async def main():
    # Example queries to test the system

    print("Nike Product Information Assistant")
    print("---------------------------------")

    print("\n\nInteractive Mode (type 'exit' to quit)")
    print("---------------------------------")

    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() == 'exit':
            break

        result = await Runner.run(nike_triage_agent, user_query)
        print(f"\n\nRESPONSE: {result.final_output}")
        print("Agent Name: ", result.last_agent.name)

if __name__ == "__main__":

    asyncio.run(main())
