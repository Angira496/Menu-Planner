# Utsav Menu Planner (LangChain + LangGraph)

An agent that asks about your event (occasion, religion/community, region, diet, guests, budget,
allergies) and suggests a culturally appropriate Indian menu.

## Setup
```
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
 -> then add your GOOGLE_API_KEY (free at aistudio.google.com/app/apikey)
```

## Run
- Web UI:      `streamlit run app.py`
- Terminal:    `python main.py`
- Check graph: `python test_flow.py`  (uses a fake LLM, no API key needed)

## How it works (graph.py)
```
START -> ask_next (loops over 7 questions, uses interrupt())
      -> build_constraints   community + diet + occasion + allergy rules
      -> retrieve_dishes     filters dishes.json by rules, region and occasion
      -> generate_menu       LLM, structured output (Pydantic)
      -> validate_menu       Python check for banned ingredients
            |-- violations (max 3 tries) -> generate_menu
            `-- clean -> review (user accepts or gives feedback)
                           |-- feedback -> generate_menu
                           `-- ok -> END
```

## Files
- `graph.py`       the LangGraph agent (state, nodes, routing, rules)
- `app.py`         Streamlit UI        - `main.py` terminal UI
- `dishes.json`    dish knowledge base - `build_kb.py` rebuilds it from a readable table
- `test_flow.py`   tests the flow with a fake LLM

## Customising
- Cultural rules: edit `COMMUNITY_RULES`, `DIET_RULES`, `OCCASION_RULES` in graph.py
- More dishes: add rows in build_kb.py, then `python build_kb.py`
- Other Gemini model: change `MODEL_NAME` in `.env`. Google retires old models often; if you get a
  'model not found' error, run `python list_models.py` and use a name it prints.

## Limits
Food customs vary within every community (region, sect, family). Treat the rules as sensible
defaults and confirm important restrictions (e.g. Jain, Navratri) with the host family.

## Budget
You enter the number of guests and a total budget in rupees. The app divides it to get the budget per guest
and sets a level: Low up to Rs 300 per guest, Medium up to Rs 700, High above that
(change `LOW_MAX_PER_GUEST` / `MEDIUM_MAX_PER_GUEST` in graph.py). The AI plans a menu whose style and
number of dishes suit that per-guest amount. No prices are shown.

## If Gemini is busy (error 503)
graph.py tries the main model 3 times, then a backup model (`FALLBACK_MODEL`, default gemini-3.6-flash).
If Google is still down, the web app shows a "Try again" button and keeps all your answers.


