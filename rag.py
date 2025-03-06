import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

openaikey = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = openaikey
model = "gpt-4o"
llm = ChatOpenAI(temperature=0.7, max_tokens=300)
# memory = ConversationBufferWindowMemory(k=3)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    input_key="query",
    k=4
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
    You are a highly knowledgeable and specialized legal assistant with expertise in interpreting legal documents.
    Your task is to provide accurate, actionable legal information based solely on the provided context.
    
    When responding:
    1. Reference specific sections of the provided content to support your answers
    2. Use precise legal terminology while explaining concepts in plain language
    3. Structure your response with clear headings and bullet points when appropriate
    4. If information is ambiguous or incomplete, acknowledge limitations rather than speculating
    
    QUESTION: '{query}'
    RELEVANT LEGAL CONTENT: '{context}'
    
    ANSWER:
    """).format(query=query, context=escaped)
    return prompt


template = """
    You are a sophisticated legal assistant analyzing legal documents and providing expert guidance.
    
    {chat_history}
    
    Guidelines for your response:
    - Directly cite relevant portions of the provided content to support your analysis
    - Balance technical legal precision with clear explanations for non-specialists
    - Format your response with appropriate structure (headings, paragraphs, bullet points)
    - When the provided content is insufficient, clearly indicate limitations and suggest appropriate next steps
    - Maintain a professional, authoritative tone while remaining accessible
    
    USER QUESTION: '{query}'
    RELEVANT LEGAL CONTENT: '{context}'
    
    ANALYSIS AND ANSWER:
    """
prompt_template_name = PromptTemplate(input_variables=['query', 'context'],
                                      template=template)

result_gen_chain = LLMChain(
    llm=llm, prompt=prompt_template_name, output_key="answer", memory=memory)


def generate_answer(prompt):
    answer = llm(prompt)
    return answer


while ch:
    print("-----------------------------------------------------------------------\n")
    print("What would you like to ask? Type 'exit' to quit.")

    query = input("Query: ")
    if query == "exit":
        break
    context = get_relevent_context_from_db(query)
    prompt = generate_rag_prompt(query, context)
    # answer = generate_answer(prompt)

    inputs = {"query": query, "context": context}

    try:
        answer = result_gen_chain(inputs)
        print("-----------------------------------------------------------------------\n")
        # Assuming result_gen_chain.memory is the correct way to access memory
        # print(result_gen_chain.memory)
        print("\n\n\n\n")
        print("YOU: ", query)
        print("BOT: ", answer["answer"])
        # Assuming answer contains the response in a 'content' attribute
        # print("BOT: ", answer.content)
    except ValueError as e:
        print(f"Error: {e}")
