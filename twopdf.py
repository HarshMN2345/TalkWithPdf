import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import os

# Streamlit interface
st.title("PDF Data Question Answering System")

# API key inputs
groq_api = st.text_input("Enter your Groq API key:", type="password")
google_api = st.text_input("Enter your Google API key:", type="password")

# Number of PDFs to upload
num_pdfs = st.number_input("Select the number of PDFs to upload:", min_value=1, value=1, step=1)

# File uploader for multiple files
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if groq_api and google_api:
    # Set the API keys
    os.environ["GOOGLE_API_KEY"] = google_api
    
    # Initialize language model and embeddings
    llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=groq_api)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Function to process the PDF files
    @st.cache_resource
    def process_pdf_files(files):
        documents = []
        for file in files:
            temp_file_path = f"temp_{file.name}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file.getvalue())
            
            loader = PyPDFLoader(temp_file_path)
            documents.extend(loader.load())
            
            os.remove(temp_file_path)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(texts, embeddings)
        return vector_store
    
    if uploaded_files:
        if len(uploaded_files) > num_pdfs:
            st.warning(f"You selected {num_pdfs} PDF(s) to upload, but uploaded {len(uploaded_files)}. Only the first {num_pdfs} will be processed.")
            uploaded_files = uploaded_files[:num_pdfs]
        elif len(uploaded_files) < num_pdfs:
            st.warning(f"You selected {num_pdfs} PDF(s) to upload, but only uploaded {len(uploaded_files)}.")
        
        with st.spinner("Processing PDF files and creating embeddings..."):
            vector_store = process_pdf_files(uploaded_files)
        
        st.success("Files processed successfully!")
        
        retriever = vector_store.as_retriever()
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        Question: {input}""")
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        user_question = st.text_input("Ask a question about the PDF content:")
        if user_question:
            with st.spinner("Generating answer..."):
                response = retrieval_chain.invoke({"input": user_question})
            st.write("Answer:", response["answer"])
    else:
        st.info(f"Please upload {num_pdfs} PDF file(s) to start.")
else:
    st.warning("Please enter both Groq and Google API keys to proceed.")