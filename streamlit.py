
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
import operator
#from agents import supervisior_agent, llm_agent, rag_agent, web_crawler_agent,validation_node
#from nodes import router
import streamlit as st
from agents import Supervisor_node, router,  LLM_node,RAG_node,WebCrawler_node,Validation_node

# Suppress Streamlit watcher error from torch.classes
import types
import torch

try:
    if hasattr(torch, 'classes'):
        torch.classes.__path__ = types.SimpleNamespace(_path=[])
except Exception:
    pass

st.title("Multi Agent System")
input_txt = st.text_input("Please enter your query here...")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("SuperVisor", Supervisor_node)
workflow.add_node("LLM", LLM_node)
workflow.add_node("RAG", RAG_node)
workflow.add_node("WebCrawler", WebCrawler_node)
workflow.add_node("Validation", Validation_node)

workflow.set_entry_point("SuperVisor")

# Add Edges
workflow.add_conditional_edges(
    "SuperVisor",
    router,
    {
        "RAG Call": "RAG",
        "LLM Call": "LLM",
        "WebCrawler Call": "WebCrawler"
    }
)

workflow.add_edge("LLM", "Validation")
workflow.add_edge("RAG", "Validation")
workflow.add_edge("WebCrawler", "Validation")
workflow.add_edge("Validation", END)
workflow.set_finish_point("Validation")


app = workflow.compile()

if input_txt:
    #state = {"messages": [HumanMessage(content=input_txt)]}
    state = {"messages":[input_txt]}
    
    result = app.invoke(state)
    st.write(result)
    
    #final_msg=st.write(result["messages"][-1])
    #final_text = final_msg.content if isinstance(final_msg, BaseMessage) else final_msg
    #st.write(final_text)

    

    
