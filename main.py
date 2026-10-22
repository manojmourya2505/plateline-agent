"""
Main file: wires the nodes into a graph and runs it.

The graph is a loop:
   agent_reason -> (model asked for a tool?) -> act -> agent_reason ...
                -> (no tool needed)           -> END

Run it with:
   .venv/bin/python main.py "Which papers do I need to sell my car?"
   .venv/bin/python main.py "Check the car with number plate MH12AB1234"
"""

import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessagesState, StateGraph

from nodes import act, agent_reason

load_dotenv(override=True)  # .env wins over any old key exported in the terminal

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1  # the most recent message


def should_continue(messages_state: MessagesState) -> str:
    """After the model replies: stop if it made no tool call, otherwise run the tool."""
    if not messages_state["messages"][LAST].tool_calls:
        return END
    return ACT


flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, agent_reason)
flow.add_node(ACT, act)
flow.set_entry_point(AGENT_REASON)

flow.add_conditional_edges(AGENT_REASON, should_continue, {END: END, ACT: ACT})
flow.add_edge(ACT, AGENT_REASON)  # after a tool runs, let the model read the result

app = flow.compile()


if __name__ == "__main__":
    # Whatever you type after main.py is the question for Lane.
    question = " ".join(sys.argv[1:]) or "Check the car with number plate MH12AB1234."
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    print(result["messages"][LAST].content)
