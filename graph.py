"""Indian occasion menu planner built with LangGraph."""
import json
import math
import os
import re
import time
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# ---------------------------------------------------------------
# 0. SETTINGS you can change
# ---------------------------------------------------------------
# Budget level, worked out per guest (total budget / number of guests).
LOW_MAX_PER_GUEST = 300      # up to Rs 300 per guest      -> Low
MEDIUM_MAX_PER_GUEST = 700   # Rs 301 to Rs 700 per guest  -> Medium; above that -> High
MAX_NUMBER = 100_000_000     # biggest number accepted for guests / budget
MAX_TEXT = 300               # longest text accepted from the user
COSTLY_FOR_LOW = ["mutton", "prawn", "saffron", "cashew", "almond", "pistachio"]  # left out of Low-budget menus

# ---------------------------------------------------------------
# 1. QUESTIONS the agent asks (key, question, options or None)
# ---------------------------------------------------------------
QUESTIONS = [
    ("occasion", "What is the occasion?",
     ["Wedding", "Birthday", "Diwali", "Eid", "Christmas", "Puja/Satyanarayan", "Navratri/Fasting",
      "Housewarming", "Other"]),
    ("community", "Which religion/community is the host family?",
     ["Hindu", "Muslim", "Sikh", "Christian", "Jain", "Parsi", "Buddhist", "Mixed"]),
    ("region", "Which regional cuisine do you prefer?",
     ["North Indian", "South Indian", "Bengali", "Gujarati", "Maharashtrian",
      "Punjabi", "Hyderabadi", "Mixed"]),
    ("diet", "Diet preference for the menu?",
     ["Vegetarian", "Sattvic (no onion/garlic)", "Eggetarian", "Non-vegetarian", "Mixed"]),
    ("guests", "How many guests will be eating? (number)", None),
    ("budget", f"What is your TOTAL food budget in rupees (₹)? Example: 5000. "
               f"Per guest: Low = up to ₹{LOW_MAX_PER_GUEST}, Medium = up to ₹{MEDIUM_MAX_PER_GUEST}, "
               f"High = above that", None),
    ("allergies", "Any allergies or foods to avoid? (comma-separated, or 'none')", None),
]

# ---------------------------------------------------------------
# 2. CULTURAL / DIETARY RULES (deterministic guardrails)
#    These are starting points: edit them with your own knowledge!
# ---------------------------------------------------------------
MEAT = ["chicken", "mutton", "lamb", "goat", "beef", "pork", "fish", "prawn",
        "shrimp", "crab", "meat", "bacon", "ham", "keema"]
ROOTS = ["onion", "garlic", "potato", "carrot", "radish", "beetroot", "turnip",
         "sweet potato", "ginger"]

COMMUNITY_RULES = {
    "jain": (["Strictly vegetarian; avoid onion, garlic and root vegetables"],
             MEAT + ["egg"] + ROOTS),
    "muslim": (["All food must be halal; no pork and no alcohol"],
               ["pork", "bacon", "ham", "lard", "wine", "rum", "beer", "whisky"]),
    "hindu": (["Avoid beef; for religious occasions prefer vegetarian"], ["beef"]),
    "sikh": (["Avoid halal/kutha meat; langar-style vegetarian is also respectful"], []),
    "christian": (["Festive, celebratory dishes; check guests' dietary needs"], []),
    "parsi": (["Parsi festive dishes like dhansak/patra ni machhi are welcome"], []),
    "buddhist": (["Prefer vegetarian or light dishes"], []),
    "mixed": (["Guests are mixed: offer clearly separated veg and non-veg options; avoid beef and pork"],
              ["beef", "pork", "bacon", "ham"]),
}

DIET_RULES = {
    "vegetarian": (["Fully vegetarian (no egg)"], MEAT + ["egg"]),
    "sattvic (no onion/garlic)": (["Sattvic: vegetarian, no onion, no garlic"],
                                  MEAT + ["egg", "onion", "garlic"]),
    "eggetarian": (["Vegetarian plus egg allowed"], MEAT),
    "non-vegetarian": (["Non-veg allowed, include a few veg dishes too"], []),
    "mixed": (["Provide both veg and non-veg sections"], []),
}


OCCASION_RULES = {
    "puja": (["Religious puja: pure vegetarian, no onion, no garlic; include a prasad-style sweet"],
             MEAT + ["egg", "onion", "garlic"]),
    "navratri": (["Navratri fasting menu: no grains (wheat/rice/dal), no onion, no garlic, no meat or egg; "
                  "use rock salt (sendha namak), kuttu, singhara, sabudana, barnyard millet, fruits, dairy"],
                 MEAT + ["egg", "onion", "garlic", "wheat", "maida", "semolina", "rice", "dal", "lentil"]),
    "eid": (["Eid: festive halal menu; include a sweet such as sheer khurma"],
            ["pork", "bacon", "ham", "lard", "wine", "rum", "beer", "whisky"]),
    "diwali": (["Diwali: include a good variety of mithai (sweets) and snacks"], []),
    "christmas": (["Christmas: festive menu with a cake or traditional sweet"], []),
    "wedding": (["Wedding: grand multi-course spread with a live counter or two"], []),
}

DISHES = json.loads((Path(__file__).parent / "dishes.json").read_text(encoding="utf-8"))

# Price per guest (Rs) of every dish, read from the "cost" field in dishes.json.
# To change a price, just edit that number in dishes.json.
PRICES = {d["name"].lower(): d["cost"] for d in DISHES}

NUMERIC_QUESTIONS = ("guests", "budget")

TIER_STYLE = {
    "Low": "economical and simple: everyday-style dishes, fewer courses, no costly ingredients like lots of dry fruits or seafood",
    "Medium": "balanced: a proper festive meal with starter, mains, bread/rice and dessert",
    "High": "premium and generous: richer dishes, more starters, a special sweet or live counter, quality ingredients",
}
DISH_COUNT = {"Low": (4, 6), "Medium": (6, 9), "High": (9, 14)}


def budget_tier(per_guest: float) -> str:
    if per_guest <= LOW_MAX_PER_GUEST:
        return "Low"
    if per_guest <= MEDIUM_MAX_PER_GUEST:
        return "Medium"
    return "High"


def parse_number(text: str):
    """'5000', '5,000', '₹5000', 'Rs 5k' -> number. Returns None if not a number."""
    t = text.lower().replace(",", "").replace("₹", "").replace("rs.", "").replace("rs", "").replace("inr", "").strip()
    mult = 1
    if t.endswith("k"):
        mult, t = 1000, t[:-1].strip()
    try:
        return float(t) * mult
    except ValueError:
        return None


# ---------------------------------------------------------------
# 3. STATE and OUTPUT SCHEMA
# ---------------------------------------------------------------
class MenuState(TypedDict, total=False):
    answers: dict
    constraints: list[str]
    banned: list[str]
    candidates: list[dict]
    menu: dict
    violations: list[str]
    attempts: int
    feedback: str
    error: str
    guests_n: int
    budget_total: float
    per_plate: float
    budget_tier: str


class Dish(BaseModel):
    name: str
    course: str = Field(description="Welcome drink, Starter, Main, Bread, Rice, Dessert, etc.")
    ingredients: list[str] = Field(description="Main ingredients, lowercase, singular")
    is_vegetarian: bool
    note: str = Field(description="One short sentence on why it suits the occasion/culture")


class Menu(BaseModel):
    title: str
    dishes: list[Dish]
    serving_tips: str


# ---------------------------------------------------------------
# Talking to Gemini (with retries, because Google's servers are sometimes busy)
# ---------------------------------------------------------------
def get_llm(model=None):
    """Gemini model. Needs GOOGLE_API_KEY in your .env file."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model or os.getenv("MODEL_NAME", "gemini-3.5-flash-lite"),
                                  timeout=30, max_retries=1)


def is_busy_error(e: Exception) -> bool:
    text = str(e).upper()
    return any(x in text for x in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "OVERLOADED", "TIMEOUT", "TIMED OUT", "504", "DEADLINE"))


def ask_gemini(prompt: str) -> Menu:
    """Try the main model 3 times, then a backup model 3 times, waiting a little between tries."""
    models = [os.getenv("MODEL_NAME", "gemini-3.5-flash-lite"), os.getenv("FALLBACK_MODEL", "gemini-3.8-flash")]
    last = None
    for model in models:
        for attempt in range(3):
            try:
                return get_llm(model).with_structured_output(Menu).invoke(prompt)
            except Exception as e:      # noqa: BLE001
                if not is_busy_error(e):
                    raise               # a real problem (wrong key, wrong model name...): show it
                last = e
                time.sleep(1 * (attempt + 1))
    raise last


# ---------------------------------------------------------------
# 4. NODES
# ---------------------------------------------------------------
def ask_next(state: MenuState):
    """Ask the next unanswered question. interrupt() pauses the graph here."""
    answers = state.get("answers", {})
    key, question, options = next(q for q in QUESTIONS if q[0] not in answers)
    reply = interrupt({"type": "question", "question": question, "options": options,
                       "error": state.get("error", "")})
    reply = str(reply).strip()
    if key in NUMERIC_QUESTIONS:
        n = parse_number(reply)
        if n is None or not math.isfinite(n) or n <= 0 or n > MAX_NUMBER:
            return {"error": "Please enter a valid number, for example 5000"}  # asks the same question again
        reply = str(int(round(n)))
    else:
        reply = reply[:MAX_TEXT]
    return {"answers": {**answers, key: reply}, "error": ""}


def occasion_token(occasion: str) -> str:
    return occasion.lower().split("/")[0].strip()


def build_constraints(state: MenuState):
    a = state["answers"]
    constraints, banned = [], []
    for rules, key in ((COMMUNITY_RULES, a["community"]), (DIET_RULES, a["diet"]),
                       (OCCASION_RULES, occasion_token(a["occasion"]))):
        c, b = rules.get(key.lower(), ([], []))
        constraints += c
        banned += b
    if a["allergies"].lower() not in ("none", "no", ""):
        items = [x.strip().lower() for x in a["allergies"].split(",") if x.strip()]
        constraints.append("Must avoid: " + ", ".join(items))
        banned += items
    guests, budget = int(a["guests"]), float(a["budget"])
    pp = budget / guests
    return {"constraints": constraints, "banned": sorted(set(banned)), "attempts": 0,
            "guests_n": guests, "budget_total": budget, "per_plate": pp, "budget_tier": budget_tier(pp)}


def retrieve_dishes(state: MenuState):
    """Pick authentic candidate dishes from the knowledge base (simple RAG)."""
    a = state["answers"]
    banned = list(state["banned"])
    if state["budget_tier"] == "Low":
        banned += COSTLY_FOR_LOW
    occ, region = occasion_token(a["occasion"]), a["region"].lower()
    revising = bool(state.get("feedback"))   # when the host asks for changes, offer every allowed dish
    scored = []
    for d in DISHES:
        text = " ".join([d["name"]] + d["ingredients"]).lower()
        if any(re.search(rf"\b{re.escape(b)}s?\b", text) for b in banned):
            continue  # violates a hard rule
        score = 0
        regions = [r.lower() for r in d["regions"]]
        score += 3 if region in regions else (1 if "pan-indian" in regions or region == "mixed" else 0)
        occs = [o.lower() for o in d["occasions"]]
        score += 3 if occ in occs else (1 if "all" in occs else 0)
        if score >= 2 or revising:
            scored.append((score, d))
    scored.sort(key=lambda x: -x[0])
    per_course, picked = {}, []
    for _, d in scored:
        if revising or per_course.get(d["course"], 0) < 5:
            per_course[d["course"]] = per_course.get(d["course"], 0) + 1
            picked.append({"name": d["name"], "course": d["course"], "diet": d["diet"],
                           "price": d["cost"]})
    return {"candidates": picked if revising else picked[:30]}


def generate_menu(state: MenuState):
    a = state["answers"]
    budget, guests, pp, tier = state["budget_total"], state["guests_n"], state["per_plate"], state["budget_tier"]
    n_lo, n_hi = DISH_COUNT[tier]
    prompt = f"""You are an expert Indian catering consultant who understands regional,
religious and cultural food traditions.

Plan a menu for this event:
{a}

Hard constraints (never break these):
{state['constraints']}
Banned ingredients: {state['banned']}

BUDGET: total Rs {budget:.0f} for {guests} guests = Rs {pp:.0f} per guest ({tier} budget).
Each dish below has a fixed price per guest in Rs. The prices of all chosen dishes added together
must be at most Rs {pp:.0f}, and should use most of that budget (about 80% to 100%).
The menu style should be {TIER_STYLE[tier]}.
Use {n_lo}-{n_hi} dishes across courses (welcome drink, starters, mains, rice/bread, dessert).
Do not write any prices yourself.

Choose ONLY from this list of dishes, and copy each dish name exactly as written
(these dishes are already rule-compliant):
{state.get('candidates')}

Match the occasion and regional cuisine."""
    if state.get("violations"):
        prompt += f"\n\nYour previous draft broke the rules: {state['violations']}. Fix them."
    if state.get("feedback"):
        names = [d["name"] for d in state["menu"]["dishes"]]
        prompt += (f"\n\nThe host gave this feedback on the previous menu: {state['feedback']}\nPrevious dishes: {names}\n"
                   "Apply EVERY change the host asked for (add, remove, replace, more or fewer dishes). "
                   "Keep the other dishes unless a change needs them removed. The host's request overrides the "
                   "dish-count range above, but keep the total price within the budget. If the host asks for a dish "
                   "that is not in the list, use the closest dish from the list instead.")
    menu = ask_gemini(prompt).model_dump()
    # Prices come from dishes.json, never from the AI. A dish not in our list is dropped.
    dishes, skipped = [], []
    for d in menu["dishes"]:
        price = PRICES.get(d["name"].strip().lower())
        if price is not None:
            d["price"] = price
            dishes.append(d)
        else:
            skipped.append(d["name"])
    menu["dishes"] = dishes
    menu["skipped"] = skipped
    menu["cost_per_guest"] = sum(d["price"] for d in dishes)
    menu["total_cost"] = menu["cost_per_guest"] * guests
    return {"menu": menu, "attempts": state.get("attempts", 0) + 1}


def validate_menu(state: MenuState):
    """Plain Python check: catches the AI breaking a hard rule."""
    violations = []
    for dish in state["menu"]["dishes"]:
        text = " ".join([dish["name"]] + dish["ingredients"]).lower()
        for b in state["banned"]:
            if re.search(rf"\b{re.escape(b)}s?\b", text):
                violations.append(f"'{dish['name']}' contains banned item '{b}'")
    return {"violations": violations}


def review(state: MenuState):
    """Show the menu to the user and wait for approval or feedback."""
    warnings = list(state.get("violations", []))
    if state["menu"].get("skipped"):
        warnings.append("Not added because they are not in our dish and price list: "
                        + ", ".join(state["menu"]["skipped"]))
    if state["menu"]["cost_per_guest"] > state["per_plate"]:
        warnings.append(f"This menu costs Rs {state['menu']['cost_per_guest']:.0f} per guest, which is above "
                        f"your Rs {state['per_plate']:.0f} per guest budget. Ask for a cheaper menu.")
    reply = interrupt({"type": "review", "menu": state["menu"], "guests": state["guests_n"],
                       "budget": state["budget_total"], "per_plate": state["per_plate"],
                       "budget_tier": state["budget_tier"], "warnings": warnings,
                       "question": "Type 'ok' to accept, or tell me what to change"})
    if str(reply).strip().lower() in ("ok", "yes", "accept", "done"):
        return {"feedback": ""}
    feedback = str(reply)[:MAX_TEXT]
    update = {"feedback": feedback, "attempts": 0, "violations": []}
    # If the host changes the budget (e.g. "increase budget to 8000"), use the new TOTAL budget
    if "budget" in feedback.lower():
        m = re.search(r"\d[\d,]*(?:\.\d+)?\s*k?\b", feedback.lower())
        n = parse_number(m.group()) if m else None
        if n and math.isfinite(n) and 0 < n <= MAX_NUMBER:
            guests = state["guests_n"]
            per_guest_given = re.search(r"per (guest|plate|person|head)|each", feedback.lower())
            total = n * guests if per_guest_given else n
            update.update(budget_total=total, per_plate=total / guests, budget_tier=budget_tier(total / guests))
    return update


# ---------------------------------------------------------------
# 5. ROUTING (conditional edges)
# ---------------------------------------------------------------
def route_intake(state: MenuState):
    done = all(q[0] in state.get("answers", {}) for q in QUESTIONS)
    return "build_constraints" if done else "ask_next"


def route_validation(state: MenuState):
    if state["violations"] and state["attempts"] < 3:
        return "generate_menu"
    return "review"


def route_review(state: MenuState):
    return "retrieve_dishes" if state.get("feedback") else END


# ---------------------------------------------------------------
# 6. BUILD THE GRAPH
# ---------------------------------------------------------------
def build_graph():
    g = StateGraph(MenuState)
    g.add_node("ask_next", ask_next)
    g.add_node("build_constraints", build_constraints)
    g.add_node("retrieve_dishes", retrieve_dishes)
    g.add_node("generate_menu", generate_menu)
    g.add_node("validate_menu", validate_menu)
    g.add_node("review", review)

    g.add_conditional_edges(START, route_intake, ["ask_next", "build_constraints"])
    g.add_conditional_edges("ask_next", route_intake, ["ask_next", "build_constraints"])
    g.add_edge("build_constraints", "retrieve_dishes")
    g.add_edge("retrieve_dishes", "generate_menu")
    g.add_edge("generate_menu", "validate_menu")
    g.add_conditional_edges("validate_menu", route_validation, ["generate_menu", "review"])
    g.add_conditional_edges("review", route_review, ["retrieve_dishes", END])
    return g.compile(checkpointer=InMemorySaver())


def menu_to_markdown(menu: dict, info: dict) -> str:
    lines = [f"# {menu['title']}", "",
             f"Guests: {info['guests']} | Budget: Rs {info['budget']:.0f} "
             f"(Rs {info['per_plate']:.0f} per guest, {info['budget_tier']} budget)", ""]
    courses = list(dict.fromkeys(d["course"] for d in menu["dishes"]))
    for c in courses:
        lines.append(f"## {c}")
        for d in (x for x in menu["dishes"] if x["course"] == c):
            tag = "Veg" if d["is_vegetarian"] else "Non-veg"
            lines.append(f"- **{d['name']}** ({tag}) - Rs {d['price']} per guest - {d['note']}")
        lines.append("")
    lines.append(f"**Menu cost:** Rs {menu['cost_per_guest']:.0f} per guest x {info['guests']} guests "
                 f"= Rs {menu['total_cost']:.0f} (budget Rs {info['budget']:.0f})")
    lines.append("")
    lines.append(f"**Serving tips:** {menu['serving_tips']}")
    return "\n".join(lines)