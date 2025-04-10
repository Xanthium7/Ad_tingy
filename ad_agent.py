from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner, WebSearchTool
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
def get_relevant_context_from_db(query):
    context = ""
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db_nccn",
                       embedding_function=embedding_function)
    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    return context


# Agent that retrieves information from Adinfo.pdf (using our vector DB)
pdf_info_agent = Agent(
    name="Nike PDF Information Specialist",
    handoff_description="Specialist for general Nike product information",
    instructions=f"""You are a knowledgeable Nike marketing specialist.
    When asked about Nike products, provide informative responses with these guidelines:
    - Use {selling_product_info['brand_voice']} tone
    - Target your message for {selling_product_info['target_audience']}
    - Highlight key selling points like {', '.join(selling_product_info['key_selling_points'])}
    - End with the call-to-action: "{selling_product_info['call_to_action']}"
    - Keep responses concise (under 100 words)
    - Use product information from the provided context
    """,
    model="gpt-4o-mini"
)


# Agent that searches for specific product information from Nike website
website_info_agent = Agent(
    name="Nike Website Product Specialist",
    handoff_description="Specialist for specific Nike product details",
    instructions=f"""You are a Nike product specialist focused on specific product details.
    When asked about specific Nike products:
    - Provide detailed specifications about the requested product
    - Include materials, technologies, and pricing when available
    - Compare with similar products when relevant
    - Use exact product names, codes, and details from the Nike website
    - Keep the {selling_product_info['brand_voice']} tone
    - End with a personalized call-to-action
    """,
    model="gpt-4o-mini",
    tools=[WebSearchTool(),],
)


# Agent that determines which type of query (general or specific product)
query_classifier_agent = Agent(
    name="Query Classifier",
    instructions="Determine if the query is asking about a specific Nike product or general Nike information.",
    output_type=ProductQueryType,
    model="gpt-4o-mini",
)


# Triage agent to handle Nike product queries
nike_triage_agent = Agent(
    name="Nike Product Information Assistant",
    instructions="""You determine what type of Nike product information the user is seeking 
    and route to the appropriate specialist. For specific product questions, use the Website 
    Product Specialist. For general Nike information, use the PDF Information Specialist.""",
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
        print(f"RESPONSE: {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
