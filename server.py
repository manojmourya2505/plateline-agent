"""
Server: a small HTTP API around Lane, so the Plateline Android app can use it.

It runs on your Mac and listens on your Wi-Fi network (not the internet).
Every request must send the secret token in the header  X-Plateline-Token.
The API keys (RapidAPI, OpenAI) stay in .env on the Mac and are never sent to the phone.

Run it with:
   .venv/bin/uvicorn server:api --host 0.0.0.0 --port 8000
"""

import hmac
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

load_dotenv(override=True)  # .env wins over any old key exported in the terminal

from main import app as graph  # the Lane graph from main.py
from tools import estimate_price, explain_offer, lookup_vehicle, read_document

TOKEN = os.getenv("PLATELINE_API_TOKEN", "")

api = FastAPI(title="Lane API", version="0.1.0")


def check_token(token: str | None) -> None:
    """Refuse the request unless the secret token matches. Refuse everything if no token is set."""
    if not TOKEN or not token or not hmac.compare_digest(token, TOKEN):
        raise HTTPException(status_code=401, detail="Wrong or missing token")


class Turn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[Turn] = []  # earlier turns of this chat, oldest first
    context: str | None = None  # facts about this seller's car and offers, from the app


class LookupIn(BaseModel):
    plate: str = Field(..., min_length=4, max_length=20)  # spaces allowed, e.g. MH 12 AB 1234


class Deduction(BaseModel):
    label: str
    amount: int


class OfferIn(BaseModel):
    dealer: str
    offer_amount: int
    deductions: list[Deduction] = []
    conditions: list[str] = []
    valid_until: str | None = None
    valuation_low: int | None = None
    valuation_high: int | None = None


@api.get("/health")
def health() -> dict:
    return {"ok": True}


@api.post("/chat")
def chat(body: ChatIn, x_plateline_token: str | None = Header(default=None)) -> dict:
    """Ask Lane a question. Send earlier turns in history so Lane remembers the chat."""
    check_token(x_plateline_token)

    messages = []
    for turn in body.history:
        if turn.role == "assistant":
            messages.append(AIMessage(content=turn.content))
        else:
            messages.append(HumanMessage(content=turn.content))
    question = body.message
    if body.context:
        question = f"{body.context}\n\nSeller's question: {body.message}"
    messages.append(HumanMessage(content=question))

    result = graph.invoke({"messages": messages})
    return {"reply": result["messages"][-1].content}


@api.post("/lookup")
def lookup(body: LookupIn, x_plateline_token: str | None = Header(default=None)) -> dict:
    """Look up a car by number plate. Uses one RapidAPI request each time."""
    check_token(x_plateline_token)
    return lookup_vehicle.invoke({"plate": body.plate})


@api.post("/document")
async def document(
    file: UploadFile = File(...),
    x_plateline_token: str | None = Header(default=None),
) -> dict:
    """Read an RC or insurance photo. The photo is deleted from the Mac straight after reading."""
    check_token(x_plateline_token)

    suffix = Path(file.filename or "photo.jpg").suffix or ".jpg"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(await file.read())
        return read_document.invoke({"image_path": path})
    finally:
        os.remove(path)


@api.post("/offer")
def offer(body: OfferIn, x_plateline_token: str | None = Header(default=None)) -> dict:
    """Work out the net amount of a dealer's offer and the checks to make. No paid calls."""
    check_token(x_plateline_token)
    return explain_offer.invoke({
        "dealer": body.dealer,
        "offer_amount": body.offer_amount,
        "deductions": [d.model_dump() for d in body.deductions],
        "conditions": body.conditions,
        "valid_until": body.valid_until,
        "valuation_low": body.valuation_low,
        "valuation_high": body.valuation_high,
    })


class ValuationIn(BaseModel):
    make: str
    model: str
    year: int | None = None
    fuel: str | None = None
    km: int | None = None
    condition: str = "Good"


@api.post("/valuation")
def valuation(body: ValuationIn, x_plateline_token: str | None = Header(default=None)) -> dict:
    """Estimate a rupee range for a car from its model, year, km and condition. AI estimate, not live prices."""
    check_token(x_plateline_token)
    result = estimate_price.invoke({
        "make": body.make, "model": body.model, "year": body.year,
        "fuel": body.fuel, "km": body.km, "condition": body.condition,
    })
    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])
    return result
