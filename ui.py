import sys
import os
import streamlit as st
import yfinance as ytf
from dotenv import load_dotenv
from typing import TypedDict, Dict, Any, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END

# Load local environment variables
load_dotenv()

# ==========================================
# 1. STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    ticker: str
    stock_data: Dict[str, Any]
    news_context: str
    recommendation: str
    justification: str
    iterations: int

# ==========================================
# 2. CORE NODES DEFINITION
# ==========================================
def data_analyst_agent(state: AgentState) -> Dict[str, Any]:
    ticker = state["ticker"]
    stock = ytf.Ticker(ticker)
    info = stock.info
    
    # Safely look up keys that work across global stock data structures
    metrics = {
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"),
        "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "volume": info.get("volume"),
        "currency": info.get("currency", "USD")
    }
    
    current_iterations = state.get("iterations", 0)
    return {"stock_data": metrics, "iterations": current_iterations + 1}

def rag_news_agent(state: AgentState) -> Dict[str, Any]:
    # Dynamic text context matching the global stock selection
    mock_news = [
        f"Global Market Alert: Trading volume shifts noticeably for {state['ticker']}.",
        f"Financial Update: Industry analysts adjust tracking expectations regarding {state['ticker']} performance."
    ]
    
    # Initialize the vector store using standard components
    vectorstore = Chroma.from_texts(texts=mock_news, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    relevant_docs = retriever.invoke(f"sentiment on {state['ticker']}")
    
    # Fixed structural extraction bug: Access .page_content directly from document objects
    context_text = "\n".join([doc.page_content for doc in relevant_docs])
    vectorstore.delete_collection()
    
    return {"news_context": context_text}

def financial_advisor_agent(state: AgentState) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an institutional financial advisor analyzing a stock from a global market exchange. Evaluate the historical metrics and current news context to issue a simulated recommendation (Buy / Sell / Hold) along with a thorough, logical justification. Note the currency provided in the data."),
        ("user", "Global Ticker: {ticker}\nMetrics: {stock_data}\nNews Context: {news_context}")
    ])
    
    chain = prompt | llm
    res = chain.invoke(state)
    
    rec = "Hold"
    if "BUY" in res.content.upper(): rec = "Buy"
    elif "SELL" in res.content.upper(): rec = "Sell"
        
    return {"recommendation": rec, "justification": res.content}

# ==========================================
# 3. ROUTER AND COMPILATION
# ==========================================
def verification_router(state: AgentState) -> Literal["loopback", "continue"]:
    data = state.get("stock_data", {})
    if not data.get("current_price") and state.get("iterations", 0) < 2:
        return "loopback"
    return "continue"

builder = StateGraph(AgentState)
builder.add_node("DataAnalyst", data_analyst_agent)
builder.add_node("RAGNews", rag_news_agent)
builder.add_node("FinancialAdvisor", financial_advisor_agent)

builder.set_entry_point("DataAnalyst")
builder.add_edge("DataAnalyst", "RAGNews")
builder.add_edge("RAGNews", "FinancialAdvisor")
builder.add_conditional_edges(
    "FinancialAdvisor",
    verification_router,
    {"loopback": "DataAnalyst", "continue": END}
)
runtime_graph = builder.compile()

# ==========================================
# 4. STREAMLIT GLOBAL INTERFACE
# ==========================================
st.set_page_config(page_title="Global AI Stock Assistant", layout="wide")
st.title("🌐 Global AI-Powered Stock Assistant")
st.subheader("Multi-Agent LangGraph System across International Exchanges")

# Layout columns for simple interaction
col_input, col_ex = st.columns([2, 2])

with col_ex:
    exchange = st.selectbox(
        "Select Target Stock Exchange:",
        [
            "US Markets (None)", 
            "National Stock Exchange - India (.NS)", 
            "Bombay Stock Exchange - India (.BO)",
            "London Stock Exchange (.L)", 
            "Tokyo Stock Exchange (.T)",
            "Toronto Stock Exchange (.TO)"
        ]
    )

with col_input:
    # Set default examples based on exchange choice
    default_ticker = "AAPL"
    if ".NS" in exchange: default_ticker = "RELIANCE"
    elif ".BO" in exchange: default_ticker = "TCS"
    elif ".L" in exchange: default_ticker = "BP"
    
    ticker_raw = st.text_input("Enter Stock Ticker Symbol:", value=default_ticker).strip().upper()

# Append suffix matching exchange rules
suffix_map = {
    "National Stock Exchange - India (.NS)": ".NS",
    "Bombay Stock Exchange - India (.BO)": ".BO",
    "London Stock Exchange (.L)": ".L",
    "Tokyo Stock Exchange (.T)": ".T",
    "Toronto Stock Exchange (.TO)": ".TO",
    "US Markets (None)": ""
}
final_ticker = f"{ticker_raw}{suffix_map[exchange]}"

if st.button("Run Multi-Agent Analysis Pipeline", type="primary"):
    with st.spinner(f"Routing agents through global graph lifecycle for {final_ticker}..."):
        try:
            res_dict = runtime_graph.invoke({"ticker": final_ticker, "iterations": 0})
            
            # Metric Card Display
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Global Ticker Symbol", res_dict["ticker"])
            with c2:
                price = res_dict["stock_data"].get("current_price")
                currency = res_dict["stock_data"].get("currency", "Units")
                st.metric("Latest Market Price", f"{price} {currency}" if price else "Data Unavailable")
            with c3:
                st.metric("Agent Action Recommendation", res_dict["recommendation"].upper())
                
            st.success("Workflow completed successfully!")
            
            st.write("### 📜 RAG-Retrieved Global News Sentiment Context")
            st.info(res_dict["news_context"])
            
            st.write("### 🔬 Multi-Agent Analytical Justification Report")
            st.markdown(res_dict["justification"])
            
        except Exception as e:
            st.error(f"Pipeline Execution Failure: {str(e)}")