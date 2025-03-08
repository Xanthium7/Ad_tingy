from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
import bs4

urls = [
    "https://www.nike.com/in/w/mens-shoes-nik1zy7ok",
    "https://www.nike.com/in/w/womens-shoes-5e1x6zy7ok",
    "https://www.nike.com/in/w/kids-shoes-1gdj0zy7ok",


]

# Load PDF documents
pdf_loaders = [PyPDFLoader('./Adinfo.pdf')]
docs = []

for file in pdf_loaders:
    docs.extend(file.load())

# Load web content from multiple URLs
for url in urls:
    web_loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("product-card__title", "product-card__price",
                        "product-card__description")
            )
        ),
    )
    web_docs = web_loader.load()
    print(web_docs)
    docs.extend(web_docs)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

# Generate embeddings
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}
)
vectorstore = Chroma.from_documents(
    docs, embedding_function, persist_directory="./chroma_db_nccn"
)

print(vectorstore._collection.count())
