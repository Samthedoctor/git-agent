import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import operator

# --- Core LangChain Imports ---
from langchain_core.messages import SystemMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool

# --- CORRECT MODEL IMPORT ---
# We are importing the class for Google's models, not OpenAI's
from langchain_google_genai import ChatGoogleGenerativeAI

# --- LangGraph Imports ---
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode

# --- 1. Load Environment Variables ---
# This ensures your GOOGLE_API_KEY is loaded for the model to use
load_dotenv()

# --- 2. Define Your Tools ---
# LangChain's @tool decorator is a modern way to define tools.
@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide a and b."""
    return a / b

tools = [add, multiply, divide]

# --- 3. Define the LLM with Bound Tools ---
# *** THIS IS THE MAIN FIX ***
# We are now using ChatGoogleGenerativeAI instead of ChatOpenAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm_with_tools = llm.bind_tools(tools)

# --- 4. Define the Graph State ---
# MessagesState is a convenient pre-built state for chat-like agents
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# --- 5. Define Graph Nodes ---
# The primary "assistant" node that calls the LLM
def assistant_node(state: AgentState):
    """Calls the LLM with the current state to decide the next action."""
    print("---NODE: Assistant---")
    # Invoke the LLM with the current list of messages
    response = llm_with_tools.invoke(state["messages"])
    # Return a dictionary to update the state with the new message
    return {"messages": [response]}

# The node for executing tools
tool_node = ToolNode(tools)

# --- 6. Define Conditional Edges ---
# This function decides whether to call a tool or end the graph
def tools_condition(state: AgentState):
    """
    Checks the latest message from the assistant.
    If it has tool calls, route to the 'tools' node.
    Otherwise, route to the 'END' node.
    """
    print("---CONDITION: Checking for Tool Calls---")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# --- 7. Build and Compile the Graph ---
builder = StateGraph(AgentState)

builder.add_node("assistant", assistant_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile()
print("Graph compiled successfully!")

# --- 8. Run the Graph ---
if __name__ == "__main__":
    print("\n--- Running Graph ---")
    # Use .stream() to see the flow of the graph step-by-step
    inputs = [HumanMessage(content="What is the result of multiplying 12 by 5, and then adding 15 to that?")]
    for event in graph.stream({"messages": inputs}):
        print("\n", event)
    print("--- Graph Finished ---")
