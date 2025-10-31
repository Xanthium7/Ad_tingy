from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner, WebSearchTool, function_tool
from pydantic import BaseModel
import asyncio
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

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
    "call_to_action": "Shop now at www.nike.com",
    "location": {
        "country": "IN",
        "country_name": "India",
        "region_code": "in"
    }
}

# Additional brand configuration for Vegeta (seasoning & culinary brand)
vegeta_product_info = {
    "brand": "Vegeta",
    "product_category": "Culinary seasonings, spice blends, marinades, and recipe inspiration",
    "website": "vegeta.com/en",
    "target_audience": "Home cooks and food enthusiasts seeking flavor and convenience",
    "brand_voice": "Flavorful, encouraging, kitchen-inspiring",
    "key_selling_points": [
        "Balanced flavor profiles",
        "High-quality ingredients",
        "Versatile culinary uses",
        "Trusted heritage",
        "Supports creative cooking"
    ],
    "call_to_action": "Discover new flavor ideas at vegeta.com/en",
    "location": {
        "country": "Global",
        "country_name": "Global",
        "region_code": "en"
    }
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
    vector_db = Chroma(persist_directory="./vegeta/chroma_db_nccn",
                       embedding_function=embedding_function)

    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    # print("CONTEXT: ", context)
    return context


# Vegeta-specific context retriever (currently same vector store path). If you later
# separate vector stores per brand, point this to a new directory.
@function_tool
def get_vegeta_context(query: str) -> str:
    """Retrieve Vegeta brand / product / recipe context from the Vegeta crawl vector DB."""
    context = ""
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./vegeta/chroma_db_nccn",
                       embedding_function=embedding_function)
    results = vector_db.similarity_search(query, k=6)
    for r in results:
        # Include URL inlined at top of each snippet if present in metadata for transparency
        url = r.metadata.get("url") or r.metadata.get("source", "")
        context += f"SOURCE: {url}\n{r.page_content}\n\n"
    return context


# Agent that retrieves information from Adinfo.pdf (using our vector DB)
pdf_info_agent = Agent(
    name="Nike PDF Information Specialist",
    handoff_description="Specialist for general Nike brand information",
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
    model="gpt-4o",
    tools=[get_relevant_context_from_db],
)


website_info_agent = Agent(
    name="Nike Website Product Specialist",
    handoff_description="Specialist for specific Nike product details",
    instructions=f"""You are a {selling_product_info['brand']} product specialist focused on specific product details.
    When asked about specific {selling_product_info['brand']} products:
    - Provide detailed specifications about the requested product
    - Include materials, technologies, and pricing when available
    - Compare with similar products when relevant
    - Use exact product names and codes from the {selling_product_info['brand']} website
    - Consistently maintain the {selling_product_info['brand_voice']} tone
    - Target your communication specifically for {selling_product_info['target_audience']}
    - Emphasize relevant key selling points: {', '.join(selling_product_info['key_selling_points'])}
    - IMPORTANT: Do NOT provide direct product purchase links as they may be region-specific
    - Instead, direct users to the main Nike website (nike.com) or appropriate section (shoes, clothing, etc.)
    - For purchase inquiries, recommend visiting nike.com/{selling_product_info["location"]["region_code"]} without specific product URLs
    - End with a personalized call-to-action that includes "{selling_product_info['call_to_action']}"
    - Keep responses concise (under 300 words)
    - Only provide information that can be found on the {selling_product_info['website']} or in its subdomains
    - NEVER make up information if it's not on the {selling_product_info['website']} or its subdomains - admit when you don't have the specific information
    - If referring to a product, use the exact product name and code from the website but DO NOT include direct product links
    - If the product is not available, provide a similar product suggestion from the website
    """,
    model="gpt-4o",
    tools=[WebSearchTool(
        user_location={"type": "approximate", "country": selling_product_info["location"]["country"]}),],
)


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
    model="gpt-4o",
)


# Vegeta brand knowledge agent (uses crawled Vegeta website data)
vegeta_site_agent = Agent(
    name="Vegeta Culinary Information Specialist",
    handoff_description="Specialist for Vegeta products, seasonings, and recipes",
    instructions=f"""You are a {vegeta_product_info['brand']} culinary knowledge specialist with access to structured website context.
    Your mission: help users with {vegeta_product_info['brand']} product details, usage ideas, flavor guidance, and recipe inspiration.
    Guidelines:
    - ALWAYS ground answers strictly in the provided context chunks.
    - When referencing a product or recipe, cite the exact product/recipe name as on the site.
    - If multiple variants exist (e.g., Natur, Original, Grill), clarify differences succinctly.
    - Emphasize these key advantages: {', '.join(vegeta_product_info['key_selling_points'])}.
    - Tone: {vegeta_product_info['brand_voice']} — practical, encouraging, flavor-forward.
    - If user asks for substitutions or unavailable info, be transparent and suggest checking vegeta.com/en for latest updates.
    - MANDATORY: After the main answer, add a section titled 'Sources:' and list each DISTINCT SOURCE URL you relied on (one per line). Do not invent URLs.
    - After citing sources, include a short Vegeta promotional note that ties directly to the users question (e.g., highlight health benefits for nutrition questions, flavor versatility for recipe questions).
    - Keep responses concise (< 220 words) unless the user explicitly asks for a deep dive.
    - NEVER fabricate nutritional data, ingredient percentages, or undisclosed proprietary details.
    - If the query is outside Vegeta scope, politely state that and redirect toward culinary usage questions.
    End every answer with the call-to-action: "{vegeta_product_info['call_to_action']}".
    """,
    model="gpt-4o",
    tools=[get_vegeta_context],
)


async def main():
    # Vegeta-only interactive assistant
    print("Vegeta Culinary Information Assistant")
    print("--------------------------------------")
    print("Ask about Vegeta products, variants, usage tips, or recipe inspiration.")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nYour Vegeta question: ")
        if user_query.lower() == 'exit':
            break
        result = await Runner.run(vegeta_site_agent, user_query.strip())
        print(f"\nRESPONSE: {result.final_output}")
        print("Agent Name:", result.last_agent.name)

if __name__ == "__main__":

    asyncio.run(main())
