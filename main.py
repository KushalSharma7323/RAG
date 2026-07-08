from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader,PyPDFLoader,WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


data2=PyPDFLoader("RAG/document_loaders/keip108.pdf")
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200

)


docs2=data2.load()
docs=splitter.split_documents(docs2)

embedding_model = HuggingFaceEmbeddings()

vector_store = Chroma.from_documents(
    documents=docs,
    persist_directory = "chroma_db",
    embedding = embedding_model
)

retriever = vector_store.as_retriever(
    search_type = "mmr", 
    search_kwargs ={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5#(more diverse result) 0--------------1 (less diverse result)
    }
)

llm = ChatMistralAI(model = "mistral-small-2506")


prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant  use easy proffesional language prefer hinglish words wherever possible and summarize the text in few bullet points. Also the give some pnemonics to remember the topic."),

        ("human","""Context  : 
         {context}
         
         Question :{question}""")
    ]
)
print ("\n\nrag system created\n\n")


while True:
    query = input("YOU:\t")
    if query=="exit":
        print("\nGoodbye Dear..\n")
        break
    docs =retriever.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    final_prompt =prompt.invoke({
        "context":context,
        "question":query

    })
    response = llm.invoke(final_prompt)
    print(f"\nChatbot:\t{response.content}\n")