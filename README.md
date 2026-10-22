# Plateline Agent

**Lane** is the AI assistant for **Plateline**, a concept app that helps Indian car owners sell their cars.

This repository holds the agent part of the project. It's a LangGraph agent that can look up a car by its number plate, read a photo of an RC, explain a dealer's offer, and answer selling questions. The Android app will call this agent later. For now it runs on a Mac, from the command line.

> Plateline is an unofficial concept project for learning and portfolio use. It uses only test or fictional data and isn't affiliated with any company.

---

## What the project does

| Feature | What the seller gets |
|---|---|
| **Plate lookup** | Enter a number plate and get the car's make, model, fuel, colour, registration date, insurance expiry and RC status. |
| **Document reading** | Photograph an RC or insurance paper. Lane reads the plate and details, so the seller doesn't have to type them. |
| **Offer explanation** | Enter a dealer's offer. Lane works out the net amount after deductions, compares it with the car's valuation range, and lists the conditions and the expiry date. |
| **Selling help** | Ask how selling works, what it costs, which papers are needed, or how long it takes. Lane answers only from an approved list. |

Lane never tells a seller whether an offer is good or bad, never gives financial advice, and never invents a fee or price.

---

## How it works

Lane is a **LangGraph** graph with two nodes:

```
            ┌──────────────┐   the model asks for a tool   ┌──────────┐
  question →│ agent_reason │ ────────────────────────────→ │   act    │
            │ (AI model)   │ ←──────────────────────────── │ (tools)  │
            └──────────────┘        tool result            └──────────┘
                   │
                   │ no tool needed
                   ▼
                 answer
```

1. The seller's question goes to the AI model, together with Lane's rules.
2. The model decides whether it needs a tool. If it does, it picks the tool and the input.
3. The `act` node runs the tool and returns the result to the model.
4. The loop repeats until the model has enough to answer, then Lane writes its reply.

The model chooses a tool from its name and description, so every tool has a clear docstring.

---

## Project layout

```
plateline-agent/
├── main.py            # Builds the graph and runs it. Your question goes after this file name.
├── nodes.py           # The two nodes (the AI model and the tool runner) and Lane's rules.
├── tools.py           # The four tools Lane can use.
├── selling_info.json  # The approved selling facts Lane can quote. Edit this, not the code.
├── test_rc_sample.png # A synthetic test image, labelled "not a real document".
├── pyproject.toml     # Project name and dependencies.
├── .python-version    # Python 3.14.
├── .env               # Your keys. Never committed (see .gitignore).
└── .gitignore         # Keeps secrets, virtual environments and caches out of Git.
```

### The tools (`tools.py`)

| Tool | Calls | Returns | Notes |
|---|---|---|---|
| `lookup_vehicle(plate)` | RapidAPI "Vehicle RC Information" | Make, model, fuel, colour, registration date, vehicle class, insurance expiry, RC status, registration authority | Uses a metered API with a limited request quota. Owner name, chassis and engine numbers are dropped. |
| `get_selling_info(topic)` | Reads `selling_info.json` | The approved answer for `how_it_works`, `papers`, `fees` or `timeline` | No internet needed. Fees and timelines are marked TO BE CONFIRMED until the business decides them. |
| `read_document(image_path)` | OpenAI vision model | Plate, make, model, fuel, registration date and insurance expiry from a photo | Owner name, address, chassis and engine numbers are never returned. Each photo is billed by OpenAI. |
| `explain_offer(...)` | None (pure arithmetic) | Net amount after deductions, a comparison with the valuation range, the conditions and the expiry date | The maths runs in code, so the numbers don't depend on the AI. |

### The steps (`nodes.py`)

- **`agent_reason`** calls the OpenAI chat model. It gets Lane's rules and the tools, and returns either an answer or a tool request.
- **`act`** is LangGraph's ready-made tool runner. It runs whichever tool the model asked for.

---

## Setup

You need a Mac with Python 3.14, a RapidAPI account subscribed to the vehicle RC API, and an OpenAI API key.

```bash
cd "/Volumes/Extreme SSD/plateline-agent"
python3 -m venv .venv
.venv/bin/pip install httpx langchain-openai langgraph python-dotenv pillow
```

Then open `.env` and fill in:

| Key | Where to get it | Needed for |
|---|---|---|
| `RAPIDAPI_KEY` | RapidAPI → your app → Authorization | Plate lookup |
| `RAPIDAPI_HOST` | Leave as `vehicle-rc-information.p.rapidapi.com` | Plate lookup |
| `OPENAI_API_KEY` | platform.openai.com | Lane's replies and document reading |
| `OPENAI_MODEL` | Leave as `gpt-4o-mini` | Lane's replies |

`.env` values override old keys in your Terminal session, because the code loads it with `override=True`. Keep `.env` private. Don't paste keys into chat, screenshots or code.

---

## Running it

Put your question after `main.py`:

```bash
.venv/bin/python main.py "Which papers do I need to sell my car?"
.venv/bin/python main.py "What fees do I pay?"
.venv/bin/python main.py "A dealer offered Rs 500000 with Rs 15000 deducted. The valuation is Rs 480000 to Rs 520000. Explain it."
.venv/bin/python main.py "Read this RC: test_rc_sample.png"
.venv/bin/python main.py "Check the car with number plate MH12AB1234"
```

If you run `main.py` with no question, it uses the plate lookup example.

### Testing one tool at a time

These commands run from the project folder:

```bash
# Selling info (no paid calls)
.venv/bin/python -c "from tools import get_selling_info; print(get_selling_info.invoke({'topic': 'papers'}))"

# Offer maths (no paid calls)
.venv/bin/python -c "from tools import explain_offer; print(explain_offer.invoke({'dealer': 'Test Dealer', 'offer_amount': 500000, 'deductions': [{'label': 'Refurbishment', 'amount': 15000}], 'valuation_low': 480000, 'valuation_high': 520000}))"

# Document reading (billed by OpenAI)
.venv/bin/python -c "from tools import read_document; print(read_document.invoke({'image_path': 'test_rc_sample.png'}))"

# Plate lookup (uses one RapidAPI request from your quota)
.venv/bin/python -c "from tools import lookup_vehicle; print(lookup_vehicle.invoke({'plate': 'MH12AB1234'}))"
```

---

## Privacy and safety

- **Keys stay on the server side.** In the planned app, the Android app never holds the RapidAPI or OpenAI keys.
- **Personal data is dropped.** The owner's name, address, chassis number and engine number are never passed to the app or shown to the seller.
- **Photos go to OpenAI.** A photo of an RC can show the owner's name, so the app must tell the seller this and ask for consent before uploading.
- **Lane only quotes approved facts.** Fees and timelines come from `selling_info.json`. Anything not in that file gets an "I don't know" and a suggestion to contact support.
- **No financial advice.** Lane describes an offer's numbers and suggests questions to ask. It doesn't say whether to accept.
- **Test data only.** `test_rc_sample.png` is a made-up image. Use only your own car's papers for real tests.

---

## Limits and known issues

- **The lookup is metered.** Requests count against your RapidAPI plan. A 403 means the key isn't subscribed to the API, and a 429 means too many requests in a short time. The agent reports both as errors.
- **Photos can be misread.** Blurry or dark images may return wrong details. The seller must check the plate before relying on it.
- **Selling facts are placeholders.** Plateline's own fee and the handover timeline are marked TO BE CONFIRMED.
- **No valuation engine yet.** The app's price range is still demo data. Lane can explain a range it's given, but it can't produce one.
- **Runs on one Mac.** There's no server or app connection yet.

---

## Roadmap

1. Wrap the agent in a small FastAPI service, running on the Mac for now.
2. Connect the Android app's Lane chat, document upload, offer check and valuation screens to that service.
3. Save each seller's results to Firestore, so they show on their home screen.
4. Add a real price estimate from sales data.
5. Add a login check, so only signed-in sellers can use the service.

---

## Credits

- Vehicle data: the RapidAPI "Vehicle RC Information" API.
- AI: OpenAI models, through LangChain.
- Agent framework: LangGraph.
