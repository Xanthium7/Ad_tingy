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
    You are a highly knowledgeable and skilled lawyer bot that reads legal documents and answers users' law-based queries. 
    Your responses should provide relevant solutions and legal advice based on the content of the provided. 
    Ensure your answers are clear, precise, and use legal terminology appropriately. 
    However, remember that your audience may not have a legal background, so explain legal concepts in an understandable manner. 
    If the PDF content does not contain information relevant to the query, suggest alternative legal resources or advise seeking professional legal counsel.
                    QUESTION: '{query}'
                    CONTENT: '{context}'
                
                ANSWER:
                """).format(query=query, context=escaped)
    return prompt


template = prompt = """
    As an expert legal advisor (A lawyer Bot), your role is to analyze legal documents and provide insightful responses to law-related questions from users. Below is the conversation history:
{chat_history}
In your response, focus on delivering actionable legal advice and solutions derived from the provided content. Your expertise should be evident through the use of precise legal terminology, yet it's crucial to ensure your explanations are accessible to those without a legal background. Simplify complex legal concepts without sacrificing accuracy.
For queries where the provided  content does not offer relevant information, guide the user towards alternative legal resources or recommend consulting a professional legal advisor for personalized counsel.
                    QUESTION: '{query}'
                    CONTENT: '{context}'
                
                ANSWER:
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
