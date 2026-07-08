from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("RAG/document_loaders/Research_paper.pdf")

docs = data.load()

splitter = TokenTextSplitter(chunk_size=1000, 
                             chunk_overlap=0)

chunks = splitter.split_documents(docs)

print(len(chunks))
for i in range(len(chunks)):
    print("chunk",i+1,"\n\n\n\n")
    print(chunks[i].page_content)
    