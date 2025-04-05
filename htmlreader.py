import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredHTMLLoader

load_dotenv()

# Load Google API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Load the HTML data
def load_html_data(file_path):
    try:
        loader = UnstructuredHTMLLoader(file_path)
        data = loader.load()
        return data[0].page_content  # Assuming data is a list and extracting the page content
    except Exception as e:
        st.error(f"Error loading HTML file: {e}")
        return ""


# Split text into chunks
def get_text_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(data)
    return chunks


# Create a FAISS vector store
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("Vector store created successfully.")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")


# Load the question-answering conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say "Answer is not available in the context." Do not provide a wrong answer.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {e}")
        return None


# Process user input and return response
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        if chain:
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"Error processing user question: {e}")


# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Chat with HTML Boat", layout="wide")
    st.header("Chat with HTML Data")

    user_question = st.text_input("Ask a Question from the HTML File")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        html_file = st.file_uploader("Upload your HTML File and Click on the Submit & Process Button", type="html")
        if st.button("Start"):
            if html_file:
                with st.spinner("Processing..."):
                    # Save uploaded file temporarily
                    with open("uploaded_file.html", "wb") as f:
                        f.write(html_file.getbuffer())

                    # Load the file using its saved path
                    html_data = load_html_data("uploaded_file.html")
                    if html_data:
                        # Split the HTML data into chunks
                        text_chunks = get_text_chunks(html_data)
                        # Create and save the vector store
                        get_vector_store(text_chunks)
            else:
                st.warning("Please upload an HTML file.")


if __name__ == "__main__":
    main()
