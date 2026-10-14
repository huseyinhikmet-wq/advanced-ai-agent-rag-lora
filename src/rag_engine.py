from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def setup_rag_system(documents_text, persist_directory="./chroma_db"):
    print(">> RAG Sistemi Kuruluyor...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([documents_text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    print(">> RAG Sistemi Hazır.")
    return retriever
