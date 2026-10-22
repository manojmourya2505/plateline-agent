"""
Tools: the actions the agent is allowed to take.

A tool is a normal Python function with the @tool decorator.
The model reads the function's name and docstring, decides when to call it,
and LangChain runs the function and gives the result back to the model.
"""

import base64
import json
import mimetypes
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv(override=True)  # .env wins over any old key exported in the terminal

# The RapidAPI host for the "Vehicle RC Information" API.
RAPID_HOST = os.getenv("RAPIDAPI_HOST", "vehicle-rc-information.p.rapidapi.com")
RAPID_URL = f"https://{RAPID_HOST}/"

# The selling facts Lane is allowed to quote. Edit selling_info.json, not this file.
SELLING_INFO_FILE = Path(__file__).parent / "selling_info.json"


@tool
def lookup_vehicle(plate: str) -> dict:
    """
    Look up an Indian vehicle by its number plate, for example KA03NU8599.
    Returns the make, model, fuel, colour, registration date, insurance date and RC status.
    If the vehicle can't be found, returns an error message instead.
    """
    if not os.getenv("RAPIDAPI_KEY") or os.getenv("RAPIDAPI_KEY", "").startswith("PASTE_"):
        return {"error": "RAPIDAPI_KEY is not set in .env"}

    # Step 1: call the API. It expects the plate in a field called VehicleNumber.
    response = httpx.post(
        RAPID_URL,
        json={"VehicleNumber": plate.upper().replace(" ", "")},
        headers={
            "X-RapidAPI-Key": os.environ["RAPIDAPI_KEY"],
            "X-RapidAPI-Host": RAPID_HOST,
        },
        timeout=30,  # stop waiting after 30 seconds
    )
    if response.status_code != 200:
        return {"error": f"RC lookup failed with HTTP {response.status_code}"}

    body = response.json()
    if not body.get("success") or not body.get("data"):
        return {"error": f"No vehicle found for {plate}"}

    d = body["data"]

    # Step 2: keep only the fields a valuation needs.
    # Left out on purpose: owner name, chassis number, engine number and financier.
    make, _, model = (d.get("makerModel") or "").partition("/")  # "BAJAJ AUTO LTD / CALIBER"
    return {
        "plate": d.get("registrationNo"),
        "make": make.strip(),
        "model": model.strip(),
        "fuel": d.get("fuelType"),
        "colour": d.get("vehicleColor"),
        "registrationDate": d.get("registrationDate"),
        "vehicleClass": d.get("vehicleClass"),
        "insuranceUpto": d.get("insuranceUpto"),
        "rcStatus": d.get("rcStatus"),
        "registrationAuthority": d.get("registrationAuthority"),
    }


@tool
def get_selling_info(topic: str) -> dict:
    """
    Answer questions about selling a car through Plateline.
    topic must be one of: how_it_works, papers, fees, timeline.
    Returns the approved answer for that topic. Only quote what this returns.
    """
    # Read the approved facts from selling_info.json, so the answers can be checked and edited.
    facts = json.loads(SELLING_INFO_FILE.read_text(encoding="utf-8"))
    if topic not in facts:
        return {"error": f"Unknown topic. Use one of: {', '.join(facts)}"}
    return {"topic": topic, "answer": facts[topic]}


@tool
def read_document(image_path: str) -> dict:
    """
    Read a photo of an Indian RC (registration certificate) or insurance paper.
    Returns the number plate, make, model, fuel, registration date and insurance expiry.
    The owner's name, address, chassis and engine numbers are never returned.
    image_path is the path to the photo file, for example rc_photo.jpg.
    """
    path = Path(image_path).expanduser()
    if not path.exists():
        return {"error": f"File not found: {image_path}"}

    # The vision model needs the image as base64 text inside the request.
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    image_b64 = base64.b64encode(path.read_bytes()).decode()

    llm = ChatOpenAI(model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"), temperature=0)
    reply = llm.invoke([
        (
            "system",
            "You read Indian vehicle documents. Return ONLY a JSON object with these keys: "
            "plate, make, model, fuel, registrationDate, insuranceUpto. "
            "Use null for anything you cannot read clearly. Never include the owner's name, "
            "address, chassis number or engine number. The plate has no spaces, e.g. MH12AB1234.",
        ),
        ("human", [
            {"type": "text", "text": "Read this document."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        ]),
    ])

    # The model sometimes wraps JSON in text or code fences, so take the part between the first { and last }.
    text = reply.content
    try:
        data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except ValueError:
        return {"error": "Could not read the document clearly. Please retake the photo in good light."}

    result = {key: data.get(key) for key in ["plate", "make", "model", "fuel", "registrationDate", "insuranceUpto"]}
    if result["plate"]:
        result["plate"] = str(result["plate"]).upper().replace(" ", "")
    return result


@tool
def explain_offer(
    dealer: str,
    offer_amount: int,
    deductions: list[dict] | None = None,
    conditions: list[str] | None = None,
    valid_until: str | None = None,
    valuation_low: int | None = None,
    valuation_high: int | None = None,
) -> dict:
    """
    Work out what a seller really gets from a dealer's offer.
    dealer: the dealer's name.
    offer_amount: the offer in rupees, as a whole number.
    deductions: list of {"label": ..., "amount": ...} items the dealer subtracts (e.g. refurbishment).
    conditions: list of conditions the offer depends on (e.g. "subject to inspection").
    valid_until: the date the offer expires, as text.
    valuation_low and valuation_high: the car's estimated range in rupees, if known.
    Returns the net amount after deductions and plain checks. It does the maths only; it doesn't judge the offer.
    """
    deductions = deductions or []
    conditions = conditions or []

    total_deductions = sum(int(item.get("amount", 0)) for item in deductions)
    net = offer_amount - total_deductions

    checks = []
    if total_deductions:
        checks.append(f"Deductions total Rs {total_deductions:,}. Ask the dealer to explain each one.")
    if valuation_low and valuation_high:
        if net < valuation_low:
            checks.append(f"The net amount is below the estimated range of Rs {valuation_low:,} to Rs {valuation_high:,}.")
        elif net > valuation_high:
            checks.append("The net amount is above the estimated range. Read the terms carefully.")
        else:
            checks.append(f"The net amount is inside the estimated range of Rs {valuation_low:,} to Rs {valuation_high:,}.")
    if conditions:
        checks.append("The offer depends on: " + "; ".join(conditions) + ".")
    if valid_until:
        checks.append(f"The offer is valid until {valid_until}.")

    return {
        "dealer": dealer,
        "offer": offer_amount,
        "net_after_deductions": net,
        "checks": checks,
    }


ESTIMATE_PROMPT = """You estimate the resale price of a used car in India, in rupees, for a private sale.
Use the make, model, registration year, fuel, kilometres driven and condition you are given.
Higher kilometres and a worse condition lower the price. Older cars are worth less each year.
Reference point: a 2015 Ford Fiesta Titanium diesel with 45,000 km in good condition sold for about Rs 2,90,000 to Rs 3,10,000.
Use the reference only to calibrate. Indian used-car prices are in lakhs for cars this age, never in thousands.
If the kilometres are unknown, assume average usage for the car's age and say so in the note.
Return ONLY a JSON object with keys: low (integer rupees), high (integer rupees), note (one short sentence).
Say it is an AI estimate, not a live listing price."""


@tool
def estimate_price(
    make: str,
    model: str,
    year: int | None = None,
    fuel: str | None = None,
    km: int | None = None,
    condition: str = "Good",
) -> dict:
    """
    Estimate the resale price of a used car in India as a rupee range (low and high).
    Use the make, model, registration year, fuel, kilometres driven and condition (Great, Good or Fair).
    This is an AI estimate, not live market data.
    """
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    facts = (
        f"Make: {make}. Model: {model}. Year: {year or 'unknown'}. Fuel: {fuel or 'unknown'}. "
        f"Kilometres driven: {km if km is not None else 'unknown'}. Condition: {condition}."
    )
    reply = llm.invoke([("system", ESTIMATE_PROMPT), ("human", facts)])
    text = reply.content
    try:
        data = json.loads(text[text.index("{"): text.rindex("}") + 1])
        low, high = int(data["low"]), int(data["high"])
    except (ValueError, KeyError, TypeError):
        return {"error": "Could not estimate the price"}
    if low <= 0 or high <= low:
        return {"error": "Could not estimate the price"}
    return {"low": low, "high": high, "note": str(data.get("note", "AI estimate"))}
