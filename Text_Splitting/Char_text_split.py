from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

splitter = CharacterTextSplitter(
   # separator="\n",
    #separator="\n\n",
    chunk_size=1000, 
    chunk_overlap=40)

data=TextLoader("RAG/document_loaders/notes.txt")

docs=data.load()
chunks = splitter.split_documents(docs)
print(len(chunks))
for i in range(len(chunks)):
    print("\n\nchunk",i+1)
    print(chunks[i].page_content)