import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
print(api_key)

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="🤓")

st.title("📄 PDF Summarizer Chatbot (Upload & Ask Questions)")

# ---------------------------
# Upload PDF
# ---------------------------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    # ---------------------------
    # Load PDF
    # ---------------------------
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # ---------------------------
    # Split text
    # ---------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(docs)

    # ---------------------------
    # Embeddings + Vector DB
    # ---------------------------
    embedding_model = HuggingFaceEmbeddings()
    
  

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory = "chroma_db"
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # ---------------------------
    # LLM
    # ---------------------------
    llm = ChatMistralAI(model="mistral-small-2506",
                        api_key=st.secrets["MISTRAL_API_KEY"])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a proffesional AI assistant .
                Your job is  to summarize the PDF provided by user in simple and easy bullet points 
                getting all the important points and providing the jist of it.
                """
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    # ---------------------------
    # Chat memory
    # ---------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---------------------------
    # Input box
    # ---------------------------
    query = st.chat_input("Ask something from your PDF...")

    if query:

        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        # Retrieve context
        docs = retriever.invoke(query)

        context = "\n\n".join([d.page_content for d in docs])

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": query
            }
        )

        with st.spinner("Thinking... 🤔"):
            response = llm.invoke(final_prompt)

        answer = response.content

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

        # Show retrieved chunks (debug mode)
        with st.expander("📌 Retrieved Context"):
            for i, d in enumerate(docs, 1):
                st.markdown(f"**Chunk {i}**")
                st.write(d.page_content)

    # cleanup temp file (optional)
    try:
        os.remove(pdf_path)
    except:
        pass


    #""" to run this code     1.streamlit run RAG/main4.py """