from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import logging
from datetime import datetime

# Configure logging
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, "app.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to extract text from various file types
def extract_text_from_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_extension == ".txt":
        return extract_text_from_text(file_path)
    else:
        # You can add handling for other file types if needed
        return ""

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_text(text_file):
    with open(text_file, "r", encoding="utf-8") as file:
        return file.read()

# Use st.cache to avoid re-computing the embeddings if file content doesn't change
@st.cache_resource
def get_embeddings_and_chain(file_contents):
    try:
        text_chunks = CharacterTextSplitter(separator="\n", chunk_size=1000,
                                            chunk_overlap=200, length_function=len).split_text(file_contents)
        
        if not text_chunks:
            raise ValueError("Text chunks are empty. Check your file content extraction.")

        embeddings = OpenAIEmbeddings()
        if not embeddings:
            raise ValueError("Embeddings are not generated correctly.")
        
        docsearch = FAISS.from_texts(text_chunks, embeddings) 
        if not docsearch:
            raise ValueError("Faiss index is not created correctly.")
        
        llm = OpenAI() 
        chain = load_qa_chain(llm, chain_type="stuff")
        
        logging.info("Embeddings and chain successfully generated.")
        return embeddings, docsearch, chain

    except ValueError as e:
        logging.error(f"Error in get_embeddings_and_chain: {e}")
        st.error(f"Error: {e}")
        return

def log_user_interaction(query, response):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"User Query: {query} - Bot Response: {response} (Timestamp: {timestamp})"
    logging.info(log_message)

# ... (previous code)

def main():
    load_dotenv()
    st.set_page_config(page_title="8Queens")
    
    st.markdown(
        '<p style="font-size: 36px;">'
        '<span style="color: red;">8</span>'
        '<span style="color: white;">Q</span>'
        '<span style="color: blue;">u</span>'
        '<span style="color: white;">e</span>'
        '<span style="color: red;">e</span>'
        '<span style="color: blue;">n</span>'
        '<span style="color: white;">s</span>'
        '</p>',
        unsafe_allow_html=True
    )

    # Specify the directory containing files
    files_directory = "files"
    
    # List all files in the directory
    files_list = [file for file in os.listdir(files_directory) if os.path.isfile(os.path.join(files_directory, file))]

    if not files_list:
        st.error(f"No files found in the '{files_directory}' directory.")
        return

    # Process all files in the directory
    for selected_file in files_list:
        # Full path to the selected file
        file_path = os.path.join(files_directory, selected_file)
        
        # extract the text
        file_text = extract_text_from_file(file_path)
        
        try:
            # Fetch embeddings and chain
            embeddings, docsearch, chain = get_embeddings_and_chain(file_text)
        except ValueError as e:
            st.error(f"Error: {e}")
            continue  # Move on to the next file

    # Initialize chat history if not present in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input above the text box
    query_key = "query_key"  # Unique key for the text input
    query = st.text_input("You (Type your question):", key=query_key)
     

    if query:
        # Bot response
        docs = docsearch.similarity_search(query, queries=[query])
        response = chain.run(input_documents=docs, question=query)

        # Log the user interaction
        log_user_interaction(query, response)

        # Add user and bot messages to chat history
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", response))

        # Clear the user input after processing
        query = ""  # Set query to an empty string to clear the input field

    # Display chat history with user and bot messages vertically aligned in descending order
    for i in range(len(st.session_state.chat_history) - 2, -1, -2):
        user_message = st.session_state.chat_history[i][1]
        bot_message = st.session_state.chat_history[i+1][1]

        # Highlight "You" and "Bot" in the messages
        user_message = user_message.replace("You", '**You**')
        bot_message = bot_message.replace("Bot", '**Bot**')

        st.write(f"You: {user_message}", key=f"user_{i}")
        st.write(f"Bot: {bot_message}", key=f"bot_{i}")

if __name__ == '__main__':
    main()
