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

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if groq_api and google_api:
    # Set the API keys
    os.environ["GOOGLE_API_KEY"] = google_api
    
    # Initialize language model and embeddings
    llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=groq_api)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Function to process the PDF file
    @st.cache_resource
    def process_pdf_file(file):
        temp_file_path = f"temp_{file.name}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.getvalue())
        
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        os.remove(temp_file_path)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(texts, embeddings)
        return vector_store
    
    if uploaded_file:
        with st.spinner("Processing PDF file and creating embeddings..."):
            vector_store = process_pdf_file(uploaded_file)
        
        st.success("File processed successfully!")
        
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
        st.info("Please upload a PDF file to start.")
else:
    st.warning("Please enter both Groq and Google API keys to proceed.")