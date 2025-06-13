from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage,AIMessage
import operator
from langchain_groq import ChatGroq
from utils import llms
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from utils.llms import embeddings
retriever=embeddings()

# model
llm = ChatGroq(model="llama-3.3-70b-versatile")

# create pydantic class
class InputCategoryParser(BaseModel):
    category: str=Field(description="selected category")
    Reasoning: str=Field(description="Reasoning behind the selected category")

# outputparser
parser = PydanticOutputParser(pydantic_object=InputCategoryParser)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],operator.add]

def Supervisor_node(state: AgentState):
    question_msg = state["messages"][-1]
    question = question_msg.content if hasattr(question_msg, "content") else question_msg

    template = """Your task is to classify the given query into one of the following categories: [USA GDP, Not Related, Real Time].
    User query: {question}
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser
    parsed_response: InputCategoryParser = chain.invoke({"question": question})

    # Optional: log reasoning
    print(f"🔎 Classified as: {parsed_response.category}")
    print(f"💡 Reasoning: {parsed_response.Reasoning}")

    return {
        "messages": [AIMessage(content=parsed_response.category)],
        "reasoning": parsed_response.Reasoning  # optional metadata
    }

def router(state: AgentState):
    category = state["messages"][-1].content if hasattr(state["messages"][-1], "content") else state["messages"][-1]

    if category == "USA GDP":
        return "RAG Call"
    elif category == "Not Related":
        return "LLM Call"
    else:
        return "WebCrawler Call"

    
def LLM_node(state:AgentState):
    # LLM function
    print("---> LLM call --->")
    question_msg = state["messages"][0]   
    question = question_msg.content if isinstance(question_msg, BaseMessage) else question_msg
    # Normal LLM call 
    
    query = "Anwer the following question with your knowlwdge of the real world. Following of the user question:" + question
    response = llm.invoke(query)
    # Return as an AIMessage
    return {
        "messages": [
            AIMessage(content=response if isinstance(response, str) else response.content)
        ]
    }
    #return {"messages": [response.content]}
    #return {"messages": [response]}
   
def format_docs(docs):
    return("\n\n".join(doc.page_content for doc in docs))

def RAG_node(state:AgentState):
    print("---> RAG call --->")

    question =state["messages"][0]

    prompt = PromptTemplate(
        template=""" You are an assistant for question-answering tasks. Use the following pices of retrieved context to answer the question.if you don't know the answer, just say you dont know. Use three sentences maximum and keep the answer concise. \n Question: {question} \n context: {context} \nAnswer:""",

        input_variables=["context","question"]
    )
    
    rag_chain =(
        {
        "context":retriever 
        | format_docs,"question":RunnablePassthrough() }
        | prompt
        | llm
        | StrOutputParser())

    result =rag_chain.invoke(question)
    return {"messages":[result]}


def get_message_content(msg):
    """Returns the content from a HumanMessage, AIMessage, or string."""
    return msg.content if isinstance(msg, BaseMessage) else msg

   
def WebCrawler_node(state:AgentState):
    # webcrawler function
    tool = TavilySearchResults()
    
    print("---> WebCrawler call --->")
    question = state["messages"][0]   # passing the first message (question) to the LLM
     
    query = "Anwer the following question with your knowlwdge of the real world. Following of the user question:" + question
    response = tool.invoke(query)
    content= []
    for r in response:
        content.append(r.get("content"))
    #return {"messages": [response]}
    return {"messages": [content]}

def Validation_node(state: AgentState) -> AgentState:
    print("---> Validation Node <----")

    user_question = state["messages"][0]
    llm_response = state["messages"][-1]

    prompt = PromptTemplate(
        template="""You are an assistant to validate the response produced by an LLM.
            Use the following response and validate it against the user query.
            If the response is helpful and correct, return YES. Otherwise, return NO.
            Question: {user_question}
            Response: {llm_response}
            """,
            
        input_variables=["user_question", "llm_response"]
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile")
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "user_question": user_question,
        "llm_response": llm_response
    })

    print(f"[Validation Result] Raw output: {result}")

    is_valid = "yes" in result.strip().lower()

    return {**state, "valid": is_valid}