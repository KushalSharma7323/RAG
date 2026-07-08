from langchain_community.retrievers import ArxivRetriever

#create a retriever 
retriever = ArxivRetriever(
    load_max_docs = 2,
    load_all_available_meta = True
)
#query arxiv
docs = retriever.invoke("what is the impact of covid-19 on engineers?")

#print results
for i,doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title",doc.metadata.get("Title"))
    print("Authors:",doc.metadata.get("Authors"))
    print("Summary:",doc.page_content[:500])
    
