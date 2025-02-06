import sqlite3
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import pdfplumber
import fitz  # PyMuPDF
import io
import time

# Database Setup
DB_PATH = 'document_store/documents.db'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# PROMPT_TEMPLATE = """
# You are an expert research assistant. Use the provided context to answer the query. 
# If unsure, state that you don't know. Be concise and factual (max 3 sentences). 

# Query: {user_query} 
# Context: {document_context} 
# Answer:
# """

PROMPT_TEMPLATE = """
Query: {user_query} 
Context: {document_context}  
Response:
"""



def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS documents (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fileName TEXT UNIQUE,
                        Data TEXT
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_query TEXT,
                        assistant_response TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        document_id INTEGER,
                        FOREIGN KEY (document_id) REFERENCES documents(Id)
                    )''')
    conn.commit()
    conn.close()

init_db()

# Save extracted text to SQLite
# def save_pdf_to_db(file_name, file_text):
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute("INSERT INTO documents (fileName, Data) VALUES (?, ?)", (file_name, file_text))
#     conn.commit()
#     conn.close()


def save_pdf_to_db(file_name, file_text):
    # Generate a timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # Format as YYYYMMDD-HHMMSS
    # Append the timestamp to the file name
    file_name_with_timestamp = f"{file_name}_{timestamp}"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO documents (fileName, Data) VALUES (?, ?)", (file_name_with_timestamp, file_text))
    conn.commit()
    conn.close()



# Load PDFs from SQLite
def load_pdfs_from_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT Id, fileName FROM documents order by Id desc")
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
# def extract_text_from_pdf(uploaded_file):
#     with pdfplumber.open(uploaded_file) as pdf:
#         text = "\n".join([page.extract_text() or "" for page in pdf.pages])
#     return text

def extract_text_from_pdf(uploaded_file):
    # Read the file as bytes
    pdf_bytes = uploaded_file.read()

    # Open the PDF using the raw bytes
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""
    
    # Loop through each page in the PDF
    for page in pdf:
        # Extract text from the page
        text += page.get_text("text")
    
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

def delete_document(doc_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE Id = ?", (doc_id,))
    conn.commit()
    conn.close()
    st.success(f"✅ Document {doc_id} deleted successfully!")

def Query(query):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    conn.commit()
    conn.close()
    return result


def delete_document_by_id(doc_id):
    result = Query(f"SELECT fileName FROM documents WHERE Id = {doc_id}")
    
    if result:  # Ensure there is a valid result
        file_name = result[0]  # Extract fileName from tuple
        confirmation = st.warning(f"⚠️ Are you sure **{file_name}**?")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅"):
                delete_document(doc_id)  # Call the delete function
                st.success(f"🗑️ Document **{file_name}** deleted successfully!")
                st.experimental_rerun()  # Refresh the page
        with col2:
            if st.button("❌"):
                st.experimental_rerun()  # Refresh without deleting
    else:
        st.error("❌ Document not found in the database.")


# List PDFs from Database
pdfs = load_pdfs_from_db()

if pdfs:
    st.subheader("Available Sources")

    options = ["Global"] + [f"{pdf[0]}: {pdf[1]}" for pdf in pdfs]
    selected_option = st.selectbox("Select a document source by Id or use 'Global'", options, index=0)

    selected_id = None if selected_option == "Global" else int(selected_option.split(':')[0])

    if selected_id:
        # Display delete icon next to document selection
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**Selected Document:** {selected_option}")
        with col2:
            if st.button("🗑️", key=f"delete_{selected_id}"):
                delete_document_by_id(selected_id)

        # Proceed only if document hasn't been deleted
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

# Save chat history to the database
def save_chat_to_db(user_query, assistant_response, document_id=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (user_query, assistant_response, document_id) VALUES (?, ?, ?)", 
                   (user_query, assistant_response, document_id))
    conn.commit()
    conn.close()

# Load chat history from the database
def load_chat_history_from_db(document_id=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if document_id:
        cursor.execute("SELECT user_query, assistant_response, timestamp FROM chat_history WHERE document_id = ? ORDER BY timestamp ASC", (document_id,))
    else:
        cursor.execute("SELECT user_query, assistant_response, timestamp FROM chat_history ORDER BY timestamp ASC")
    chat_history = cursor.fetchall()
    conn.close()
    return chat_history

# Custom CSS for chat messages
st.markdown("""
    <style>
        .user-chat {
            background-color: #4D5C6D;
            border-radius: 15px;
            padding: 10px 15px;
            margin: 10px;
            min-width: 90%;
            text-align: right;
            float: right;
            clear: both;
            color:#ffffff
        }
        .assistant-chat {
            background-color:#CED6DE;
            border-radius: 15px;
            padding: 10px 15px;
            margin: 10px;
            min-width: 90%;
            text-align: left;
            float: left;
            clear: both;
            color: #000000
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Display the chat history (optional)
if selected_id:
    chat_history = load_chat_history_from_db(document_id=selected_id)
    for query, response, timestamp in chat_history:
        st.markdown(f"**{timestamp}**")
        st.markdown(f"<div class='chat-container'><div class='user-chat'>{query}</div><div class='assistant-chat'>{response}</div></div>", unsafe_allow_html=True)

# Chat Functionality
user_input = st.chat_input("Enter your question...")
if user_input:
    with st.chat_message("user"):
        st.markdown(f"<div class='user-chat'>{user_input}</div>", unsafe_allow_html=True)
    
    with st.spinner("Generating Response..."):
        if selected_option == "Global":
            context_text = ""  # No document context needed
        
        elif selected_option == "Global":
            # If "Global" is selected, no document context is used
            context_text = ""
            st.info("🌎 Global mode activated. You will be talking directly to the LLM.")
        else:
            relevant_docs = [get_document_by_id(selected_id)]
            context_text = "\n\n".join(relevant_docs) if relevant_docs else "No relevant context found."

        conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        response_chain = conversation_prompt | LANGUAGE_MODEL
        ai_response = response_chain.invoke({"user_query": user_input, "document_context": context_text})
        
        # Save this conversation to the database
        save_chat_to_db(user_input, ai_response, selected_id)
    
    with st.chat_message("assistant", avatar="😎"):
        st.markdown(f"<div class='assistant-chat'>{ai_response}</div>", unsafe_allow_html=True)
