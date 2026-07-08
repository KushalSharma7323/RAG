from langchain_text_splitters import SemanticTextSplitter
from langchain_community.document_loaders import TextLoader,PyPDFLoader
 
data=TextLoader("RAG/document_loaders/notes.txt")
docs=data.load()

splitter = SemanticTextSplitter(
    chunk_size=100,
    chunk_overlap=20)

chunks = splitter.split_documents(docs)
print(len(chunks))
for i in range(len(chunks)):
    print("\n\nchunk",i+1)
    print(chunks[i].page_content)