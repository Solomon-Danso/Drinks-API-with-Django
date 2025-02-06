import sqlite3
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import pdfplumber

# Database Setup
DB_PATH = 'document_store/documents.db'
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

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS documents (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fileName TEXT UNIQUE,
                        Data TEXT
                    )''')
    conn.commit()
    conn.close()

init_db()

# Save extracted text to SQLite
def save_pdf_to_db(file_name, file_text):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO documents (fileName, Data) VALUES (?, ?)", (file_name, file_text))
    conn.commit()
    conn.close()

# Load PDFs from SQLite
def load_pdfs_from_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT Id, fileName FROM documents")
    pdfs = cursor.fetchall()
    conn.close()
    return pdfs

# Retrieve document data by Id
def get_document_by_id(doc_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT Data FROM documents WHERE Id = ?", (doc_id,))
    doc = cursor.fetchone()
    conn.close()
    return doc[0] if doc else None

# Extract text from PDF bytes
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    return text

# Load all document text from database
def load_all_documents_from_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT Data FROM documents")
    docs = cursor.fetchall()
    conn.close()
    return [doc[0] for doc in docs]  # Extract text only

# Process and store text chunks
def process_pdf_text(text):
    text_processor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    chunks = text_processor.split_text(text)
    DOCUMENT_VECTOR_DB.add_texts(chunks)
    return "\n\n".join(chunks)

# UI Configuration
st.title("📘 Hydot AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# List PDFs from Database
pdfs = load_pdfs_from_db()

if pdfs:
    st.subheader("Available PDFs")
    options = ["Select a document"] + [f"{pdf[0]}: {pdf[1]}" for pdf in pdfs] + ["Global"]
    selected_option = st.selectbox("Select a document by Id or choose 'Global'", options)
    
    selected_id = None if selected_option == "Select a document" or selected_option == "Global" else int(selected_option.split(':')[0])
    
    if selected_id:
        document_text = get_document_by_id(selected_id)
        if document_text:
            processed_chunks = process_pdf_text(document_text)
            st.success(f"✅ Document {selected_option} loaded successfully!")
        else:
            st.error("❌ Document not found in database.")
    
    elif selected_option == "Global":
        st.info("🌎 Global mode activated. You will be talking directly to the LLM.")

# Upload New PDF
uploaded_pdf = st.file_uploader("Upload Research Document (PDF)", type="pdf", help="Select a PDF document for analysis")
if uploaded_pdf:
    extracted_text = extract_text_from_pdf(uploaded_pdf)
    save_pdf_to_db(uploaded_pdf.name, extracted_text)
    st.success("✅ New document processed and stored in the database!")

# Chat Functionality
user_input = st.chat_input("Enter your question...")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.spinner("Analyzing document..."):
        if selected_option == "Global":
            context_text = ""  # No document context needed
        else:
            relevant_docs = load_all_documents_from_db()  # Load all stored docs
            context_text = "\n\n".join(relevant_docs) if relevant_docs else "No relevant context found."

        conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        response_chain = conversation_prompt | LANGUAGE_MODEL
        ai_response = response_chain.invoke({"user_query": user_input, "document_context": context_text})
    
    with st.chat_message("assistant", avatar="😎"):
        st.write(ai_response)
