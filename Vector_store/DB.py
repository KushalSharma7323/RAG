from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

docs=[
    Document(page_content="Python is widely used in Aritficial Intelligence.",
             metadata={"source":"AI_BOOK"}),
    Document(page_content="Pandas is used for data analysis in Python",
            metadata={"source":"Data_Science_Book"}),
    Document(page_content="Neural Networks are used in deep learning",
            metadata={"source":"DL_BOOK"})
]

embedding_model = HuggingFaceEmbeddings(
    model_name="unsloth/embeddinggemma-300m",
        )

vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory = "chroma-db"
)
result = vectorstore.similarity_search("what is used for data analysis?",k=2)
for r in result:
    print(r.page_content)
