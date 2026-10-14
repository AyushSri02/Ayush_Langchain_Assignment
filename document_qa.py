import os
import sys
import streamlit as st
from dotenv import load_dotenv

# LlamaIndex core imports
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding

# LangChain integration imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment configuration
load_dotenv()

PERSIST_DIR = "./storage"
DOCS_DIR = "./documents"

# Ensure files/storage directory layout exists securely
os.makedirs(DOCS_DIR, exist_ok=True)

@st.cache_resource
def initialize_llamaindex_engine():
    """Ingests documents using LlamaIndex and constructs/loads a semantic vector storage matrix."""
    # Configure global embedding model settings for LlamaIndex components
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    
    if os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0:
        # Load index directly if pre-computed vectors already exist
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context, embed_model=embed_model)
    else:
        # Check if documents exist before scanning directories
        if len(os.listdir(DOCS_DIR)) == 0:
            return None
        
        # LlamaIndex structural file reader parsing
        documents = SimpleDirectoryReader(DOCS_DIR).load_data()
        
        # Build Vector Store index
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        
    return index

# Initialize Index
index_instance = initialize_llamaindex_engine()

# =======================================================
# LANGCHAIN BRIDGING WORKFLOW
# =======================================================
def get_langchain_qa_chain(retriever):
    """Orchestrates LangChain prompt logic, LLM calls, and output parsing constraints."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Structural prompt template satisfying LangChain constraints
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert document analysis assistant. Answer the user query using strictly "
            "the provided retrieved document context segments. If the answers are not found within the context, "
            "state clearly that the information is missing from the uploaded source files.\n\n"
            "Retrieved Document Context:\n{context}"
        )),
        ("user", "User Query: {query}")
    ])
    
    # LangChain Expression Language (LCEL) chain sequence with StrOutputParser
    chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# =======================================================
# STREAMLIT GRAPHICAL USER INTERFACE
# =======================================================
st.set_page_config(page_title="AI Document QA System", layout="wide")
st.title("📚 Advanced AI Document QA System (RAG)")
st.subheader("Hybrid Integration Workflow: LlamaIndex Indexing + LangChain Orchestration")

# Sidebar file management module panel
with st.sidebar:
    st.header("🗂️ Data Ingestion Center")
    uploaded_files = st.file_uploader(
        "Upload reference documents (PDFs, TXT):", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(DOCS_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        
        if st.button("Process & Re-index Database", type="primary"):
            with st.spinner("LlamaIndex parsing and vector indexing document fragments..."):
                # Clear existing index cache explicitly to trigger rebuild parameters
                st.cache_resource.clear()
                if os.path.exists(PERSIST_DIR):
                    import shutil
                    shutil.rmtree(PERSIST_DIR)
                index_instance = initialize_llamaindex_engine()
                st.success("Documents successfully vectorized and stored!")

# Main Panel Interface logic layouts
if index_instance is None:
    st.info("👋 Welcome! Please upload documents in the sidebar manager panel to initialize the RAG vector engine data storage.")
else:
    # 1. Instantiate LlamaIndex Retriever Node component constraint
    base_retriever = index_instance.as_retriever(similarity_top_k=3)
    
    # 2. Bridge LlamaIndex retriever structure to yield LangChain text input context formats
    def native_llamaindex_retriever_bridge(query_str: str) -> str:
        nodes = base_retriever.retrieve(query_str)
        return "\n\n---\n\n".join([node.node.get_content() for node in nodes])

    # 3. Instantiate LangChain execution sequence
    qa_chain = get_langchain_qa_chain(native_llamaindex_retriever_bridge)
    
    st.write("### 🔍 Semantic Retrieval Engine Query Prompt")
    user_query = st.text_input("Ask a question about your uploaded context documents:")
    
    # Functional UI layout controls split options
    col_ask, col_sum = st.columns(2)
    
    with col_ask:
        if st.button("Execute Targeted Query Search", use_container_width=True, type="primary"):
            if user_query:
                with st.spinner("Querying LlamaIndex Vector Nodes & Streaming LLM response..."):
                    try:
                        # Invoke LangChain runnables pipeline sequence
                        response_text = qa_chain.invoke(user_query)
                        st.write("### 🤖 Context-Aware Response Output:")
                        st.write(response_text)
                    except Exception as e:
                        st.error(f"Execution Error: {str(e)}")
            else:
                st.warning("Please type a valid textual query query sequence.")
                
    with col_sum:
        if st.button("Generate Complete Document Summary", use_container_width=True):
            with st.spinner("Synthesizing context partitions for summary compilation..."):
                try:
                    summary_chain = get_langchain_qa_chain(native_llamaindex_retriever_bridge)
                    summary_output = summary_chain.invoke("Provide a concise executive summary of the entire document, highlighting its core themes and main takeaways.")
                    st.write("### 📋 Executive Summary Analysis:")
                    st.write(summary_output)
                except Exception as e:
                    st.error(f"Summary Compilation Error: {str(e)}")