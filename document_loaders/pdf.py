from langchain_community.document_loaders import PyPDFLoader
data = PyPDFLoader("RAG/document_loaders/Research_paper.pdf")
from langchain_mistralai import ChatMistralAI
docs = data.load()
print(len(docs))
print(docs[11])

