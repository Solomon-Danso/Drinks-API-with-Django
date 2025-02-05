import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# UI Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    /* Additional styling */
    </style>
    """, unsafe_allow_html=True)

# Constants and Setup
PDF_STORAGE_PATH = 'document_store/pdf/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# UI Configuration
st.title("📘 Hydot AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# List existing PDFs in the folder
existing_pdfs = [f for f in os.listdir(PDF_STORAGE_PATH) if f.endswith('.pdf')]
if existing_pdfs:
    st.subheader("Available PDFs")
    for i, pdf in enumerate(existing_pdfs, 1):
        st.write(f"{i}. {pdf}")
    selected_pdf_number = st.number_input(
        "Select a document by number (or 0 to upload a new one)", min_value=0, max_value=len(existing_pdfs)
    )
    
    if selected_pdf_number > 0:
        # Load the selected PDF
        selected_pdf = existing_pdfs[selected_pdf_number - 1]
        file_path = os.path.join(PDF_STORAGE_PATH, selected_pdf)
        raw_docs = load_pdf_documents(file_path)
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)
        st.success(f"✅ {selected_pdf} processed successfully!")
        
    elif selected_pdf_number == 0:
        # Upload a new PDF
        uploaded_pdf = st.file_uploader(
            "Upload Research Document (PDF)",
            type="pdf",
            help="Select a PDF document for analysis"
        )

        if uploaded_pdf:
            saved_path = save_uploaded_file(uploaded_pdf)
            raw_docs = load_pdf_documents(saved_path)
            processed_chunks = chunk_documents(raw_docs)
            index_documents(processed_chunks)
            st.success("✅ New document processed successfully!")

else:
    st.warning("No existing PDFs found. Please upload a new document.")

# Chat functionality
user_input = st.chat_input("Enter your question about the document...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.spinner("Analyzing document..."):
        relevant_docs = find_related_documents(user_input)
        ai_response = generate_answer(user_input, relevant_docs)
        
    with st.chat_message("assistant", avatar="🤖"):
        st.write(ai_response)
