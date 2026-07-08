#Load pdf
#split into chunks 
#create the embeddings 
#store into chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv 
load_dotenv()
data=PyPDFLoader(
    "RAG\document_loaders\Research_paper.pdf"
)
docs=data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=40)

chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="unsloth/embeddinggemma-300m",
        )

vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_model,
    persist_directory = "chroma_db" 
)
