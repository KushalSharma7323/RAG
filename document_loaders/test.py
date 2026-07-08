from langchain_community.document_loaders import TextLoader
data=TextLoader("RAG/document_loaders/notes.txt")
#print(data)
docs=data.load()
# print(docs)
# print(docs[0])
print(docs[0].page_content)
# print(len(docs))
