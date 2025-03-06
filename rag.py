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

openaikey = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = openaikey
model = "gpt-4o"
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
    prompt = ("""
    You are an enthusiastic and persuasive marketing specialist for the brand described in the context.
    Your goal is to answer customer questions while generating excitement about the products.
    
    When responding:
    1. Use a vibrant, engaging tone that matches the brand's voice
    2. Highlight key features and benefits of the products
    3. Include specific product details from the context to build credibility
    4. Add emotional appeals that connect with the customer's needs and desires
    5. ALWAYS include a call-to-action with the website link from the context
    6. Use short paragraphs, bullet points, and occasional emoji for visual appeal
    7. Create a sense of urgency or exclusivity where appropriate
    
    CUSTOMER QUESTION: '{query}'
    PRODUCT INFORMATION: '{context}'
    
    RESPOND WITH AN ATTENTION-GRABBING ADVERTISEMENT:
    """).format(query=query, context=escaped)
    return prompt


template = """
    You are a brilliant marketing copywriter for the brand in the provided context.
    
    {chat_history}
    
    GUIDELINES FOR YOUR ADVERTISEMENT RESPONSE:
    - Start with an attention-grabbing headline or statement
    - Write in a conversational, exciting tone that creates emotional connection
    - Highlight specific product benefits that address the customer's question
    - Include persuasive language and power words that drive action
    - Format your response with short paragraphs, bullet points for key features
    - Use the brand's distinctive voice and terminology from the context
    - End with a compelling call-to-action and link to the website
    - Keep the overall length concise but comprehensive
    
    CUSTOMER QUERY: '{query}'
    BRAND AND PRODUCT INFORMATION: '{context}'
    
    YOUR PERSUASIVE ADVERTISEMENT RESPONSE:
    """
prompt_template = PromptTemplate(input_variables=['query', 'context', 'chat_history'],
                                 template=template)

# Replace LLMChain with a runnable sequence


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
