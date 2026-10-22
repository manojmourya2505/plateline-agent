"""
Nodes: the steps of the graph.

agent_reason : the AI model reads the conversation, then either answers
               or asks for a tool to be run.
act          : runs the tool the model asked for and returns the result.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from tools import explain_offer, get_selling_info, lookup_vehicle, read_document

load_dotenv(override=True)  # .env wins over any old key exported in the terminal

# The tools the model is allowed to use.
tools = [lookup_vehicle, get_selling_info, read_document, explain_offer]  # the model picks the right one

# The model. bind_tools() tells it which tools exist, so it can ask to use them.
llm_with_tools = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,  # steady, factual answers
).bind_tools(tools)

SYSTEM_MESSAGE = """
You are Lane, the assistant inside the Plateline car-selling app. You help Indian car sellers.

Rules:
- If the user gives a number plate, call lookup_vehicle. Then reply in two short sentences,
  using only the facts it returned. If rcStatus is not 'ACTIVE', warn the seller clearly.
- If the user asks how selling works, the fees, the papers, or the timeline, call
  get_selling_info with the matching topic (how_it_works, papers, fees or timeline).
  Answer only from what it returns.
- If the user shares a photo of an RC or insurance paper (a file path), call read_document.
  Show the seller the plate and details it found, and ask them to check the plate is correct.
  Only call lookup_vehicle after that if the seller asks, or if the document had no RC status.
- If the user asks about a dealer's offer, call explain_offer with the offer details they give.
  Explain the net amount and each check in plain language. Give the facts and suggest questions
  to ask the dealer. Do not say whether the offer is good or bad, and do not give financial advice.
- If a tool has no answer, say so and suggest contacting Plateline support.
- Never invent fees, rules or prices. Do not guess a price for the car.
"""


def agent_reason(messages_state: MessagesState) -> dict:
    """Ask the model what to do next. It may answer, or request a tool call."""
    response = llm_with_tools.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *messages_state["messages"]]
    )
    return {"messages": [response]}


# A ready-made node that runs whatever tool calls the model asked for.
act = ToolNode(tools=tools)
