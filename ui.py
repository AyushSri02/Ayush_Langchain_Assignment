import os
import sys
import streamlit as st
import yfinance as ytf
from dotenv import load_dotenv
from typing import TypedDict, Dict, Any, Literal

# LangChain infrastructure imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

# LangGraph state workflow imports
from langgraph.graph import StateGraph, END

# Force load environment context keys from root folder
load_dotenv()

# ==========================================
# 1. LANGGRAPH STATE DICTIONARY DEFINITION
# ==========================================
class AgentState(TypedDict):
    ticker: str
    stock_data: Dict[str, Any]
    news_context: str
    recommendation: str
    justification: str
    iterations: int
    error_log: str

# ==========================================
# 2. MULTI-AGENT NODE DEFINITIONS
# ==========================================

def data_analyst_agent(state: AgentState) -> Dict[str, Any]:
    """
    Agent 1: Quantitative Data Extraction Node.
    Connects with global stock market exchanges to extract core tracking attributes.
    """
    ticker = state["ticker"]
    current_iterations = state.get("iterations", 0)
    
    try:
        stock = ytf.Ticker(ticker)
        info = stock.info
        
        # Safely parse financial metrics across international data schemas
        metrics = {
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"),
            "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "volume": info.get("volume"),
            "currency": info.get("currency", "USD")
        }
        return {"stock_data": metrics, "iterations": current_iterations + 1, "error_log": ""}
    except Exception as e:
        # Pass missing metrics forward to test the graph's self-healing loopback edge
        return {"stock_data": {}, "iterations": current_iterations + 1, "error_log": str(e)}


def rag_news_agent(state: AgentState) -> Dict[str, Any]:
    """
    Agent 2: Retrieval-Augmented Generation (RAG) Node.
    Vectorizes transient external market data feeds to inject context.
    """
    ticker = state["ticker"]
    
    # Dynamic document feed mirroring actual internal corporate financial streams
    mock_market_news = [
        f"Global Exchange Warning: High institutional volume shifts observed for asset {ticker}.",
        f"Financial Horizon Update: Market analysts adjust forecast benchmarks regarding {ticker} core performance metrics."
    ]
    
    # Initialize isolated transient vector database context
    vectorstore = Chroma.from_texts(texts=mock_market_news, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    # Retrieve semantic data matching ticker context criteria
    relevant_docs = retriever.invoke(f"latest market trends sentiment on {ticker}")
    context_text = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Clean vector collection memory explicitly after use
    vectorstore.delete_collection()
    
    return {"news_context": context_text}


def financial_advisor_agent(state: AgentState) -> Dict[str, Any]:
    """
    Agent 3: LLM Reasoning & Strategy Recommendation Node.
    Synthesizes numerical indicators and text news arrays to generate conclusions.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Institutional advisor system blueprint rules
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an elite institutional financial advisor analyzing a stock from a global market exchange.\n"
            "Evaluate the structural financial metrics and news context provided by your agent team to produce "
            "a formal simulated action recommendation: [BUY], [SELL], or [HOLD].\n"
            "You must provide a detailed markdown breakdown highlighting your logical analytical justification. "
            "Address any currency tags present in the payload data."
        )),
        ("user", "Global Ticker: {ticker}\nFinancial Metrics: {stock_data}\nNews Context: {news_context}")
    ])
    
    # LangChain Expression Language (LCEL) runtime execution pipeline
    chain = prompt | llm
    response = chain.invoke({
        "ticker": state["ticker"],
        "stock_data": state["stock_data"],
        "news_context": state["news_context"]
    })
    
    content = response.content
    
    # Clean output parsing matching structural classification properties
    rec = "Hold"
    if "[BUY]" in content.upper() or "BUY" in content.upper(): rec = "Buy"
    elif "[SELL]" in content.upper() or "SELL" in content.upper(): rec = "Sell"
        
    return {"recommendation": rec, "justification": content}

# ==========================================
# 3. CONDITIONAL EDGE ROUTER LOGIC
# ==========================================
def verification_router(state: AgentState) -> Literal["loopback", "continue"]:
    """
    Algorithmic Conditional Edge.
    Validates data structural health before allowing the graph state to resolve.
    """
    data = state.get("stock_data", {})
    iterations = state.get("iterations", 0)
    
    # Quality Guardrail Check: If yfinance failed to retrieve live metrics,
    # force a loopback path to re-attempt extraction (max 2 attempts).
    if (not data or not data.get("current_price")) and iterations < 2:
        return "loopback"
    return "continue"

# ==========================================
# 4. COMPOSING THE LANGGRAPH TOPOLOGY
# ==========================================
builder = StateGraph(AgentState)

# Register specialized nodes to the graph blueprint
builder.add_node("DataAnalyst", data_analyst_agent)
builder.add_node("RAGNews", rag_news_agent)
builder.add_node("FinancialAdvisor", financial_advisor_agent)

# Structure operational processing lineage links
builder.set_entry_point("DataAnalyst")
builder.add_edge("DataAnalyst", "RAGNews")
builder.add_edge("RAGNews", "FinancialAdvisor")

# Inject constraints via the Conditional Loopback Edge
builder.add_conditional_edges(
    "FinancialAdvisor",
    verification_router,
    {
        "loopback": "DataAnalyst",  # Reroute to retry data extraction
        "continue": END             # Safely terminate the state machine loop
    }
)

# Compile production runnable instance object
runtime_graph = builder.compile()

# ==========================================
# 5. STREAMLIT GRAPHICAL INTERFACE
# ==========================================
st.set_page_config(page_title="AI Stock Market Assistant", layout="wide")
st.title("📈 AI-Powered Multi-Agent Stock Assistant")
st.subheader("Autonomous Financial Workflows Powered by LangGraph & LangChain")

# Interface Interaction Configuration Column Block Layouts
col_input, col_ex = st.columns(2)

with col_ex:
    exchange = st.selectbox(
        "Select Target Stock Exchange:",
        [
            "US Market Indices (None)", 
            "National Stock Exchange - India (.NS)", 
            "Bombay Stock Exchange - India (.BO)",
            "London Stock Exchange (.L)", 
            "Tokyo Stock Exchange (.T)",
            "Toronto Stock Exchange (.TO)"
        ]
    )

with col_input:
    # Dynamically match standard ticker code placeholders based on user choice
    default_ticker = "AAPL"
    if ".NS" in exchange: default_ticker = "RELIANCE"
    elif ".BO" in exchange: default_ticker = "TCS"
    elif ".L" in exchange: default_ticker = "BP"
    elif ".TO" in exchange: default_ticker = "SHOP"
    
    ticker_raw = st.text_input("Enter Stock Ticker Symbol:", value=default_ticker).strip().upper()

# Append exchange suffixes to allow global market evaluations
suffix_map = {
    "National Stock Exchange - India (.NS)": ".NS",
    "Bombay Stock Exchange - India (.BO)": ".BO",
    "London Stock Exchange (.L)": ".L",
    "Tokyo Stock Exchange (.T)": ".T",
    "Toronto Stock Exchange (.TO)": ".TO",
    "US Market Indices (None)": ""
}
final_ticker = f"{ticker_raw}{suffix_map[exchange]}"

if st.button("Run Multi-Agent Analysis Pipeline", type="primary", use_container_width=True):
    with st.spinner(f"Routing agents through autonomous graph lifecycle for {final_ticker}..."):
        try:
            # Invoke the active graph runnable using initial dictionary parameters
            res_dict = runtime_graph.invoke({"ticker": final_ticker, "iterations": 0})
            
            # Render clear visual KPI callout cards
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Processed Global Ticker", res_dict["ticker"])
            with c2:
                price = res_dict["stock_data"].get("current_price")
                currency = res_dict["stock_data"].get("currency", "Units")
                st.metric("Live Market Valuation", f"{price} {currency}" if price else "Data N/A")
            with c3:
                st.metric("Agent Action Recommendation", res_dict["recommendation"].upper())
                
            st.success(f"LangGraph state machine verified and completed in {res_dict['iterations']} workflow cycles!")
            
            # Display outputs under explicit scannable headers
            st.write("### 📜 RAG-Retrieved Vector News Sentiment Context")
            st.info(res_dict["news_context"])
            
            st.write("### 🔬 Multi-Agent Analytical Justification Report")
            st.markdown(res_dict["justification"])
            
        except Exception as e:
            st.error(f"System Pipeline Execution Failure: {str(e)}")
