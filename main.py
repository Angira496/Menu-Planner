from langgraph.types import Command
from graph import build_graph


def show_menu(payload: dict):
    menu = payload["menu"]
    print(f"\n=== {menu['title']} ===")
    print(f"{payload['guests']} guests | Budget Rs {payload['budget']:.0f} "
          f"(Rs {payload['per_plate']:.0f} per guest, {payload['budget_tier']} budget)")
    for w in payload.get("warnings", []):
        print("WARNING:", w)
    for d in menu["dishes"]:
        tag = "(veg)" if d["is_vegetarian"] else "(non-veg)"
        print(f"- [{d['course']}] {d['name']} {tag} - Rs {d['price']} per guest")
        if d["note"]:
            print(f"    {d['note']}")
    print(f"\nMenu cost: Rs {menu['cost_per_guest']:.0f} per guest x {payload['guests']} guests "
          f"= Rs {menu['total_cost']:.0f}")
    print(f"\nTips: {menu['serving_tips']}\n")


def run_step(graph, payload, config):
    """Run the graph. If something fails (for example Gemini is busy), offer to try again."""
    while True:
        try:
            return graph.invoke(payload, config)
        except Exception as e:  # noqa: BLE001
            print("\nProblem:", e)
            if input("Try again? (y/n) > ").strip().lower() != "y":
                raise SystemExit
            payload = None      # None = continue from the last saved step


def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": "session-1"}}
    result = run_step(graph, {"answers": {}}, config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        if payload["type"] == "review":
            show_menu(payload)
        if payload.get("error"):
            print(payload["error"])
        if payload.get("options"):
            print("Options:", ", ".join(payload["options"]))
        reply = input(payload["question"] + " > ")
        result = run_step(graph, Command(resume=reply), config)

    print("Final menu confirmed. Enjoy your event!")


if __name__ == "__main__":
    main()
