import os
import json
import uuid
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from vector_store import search_by_text, get_similar_to_product, add_product
from web_search import search_product_online

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))


class SearchState(TypedDict):
    query: str
    db_results: list[dict]
    web_results: list[dict]
    web_product: dict
    suggestions: list[dict]
    answer: str
    used_web: bool
    saved_to_db: bool


def node_db_search(state: SearchState) -> SearchState:
    results = search_by_text(state["query"], top_k=8, threshold=0.35)
    return {**state, "db_results": results}


def node_decide(state: SearchState) -> str:
    return "suggest" if state["db_results"] else "web_search"


def node_web_search(state: SearchState) -> SearchState:
    results = search_product_online(state["query"])
    return {**state, "web_results": results, "used_web": True}


def node_extract_and_match(state: SearchState) -> SearchState:
    web_text = "\n\n".join(
        f"Title: {r['title']}\n{r['content'][:600]}"
        for r in state["web_results"]
    )

    prompt = f"""Ψάχνουμε πληροφορίες για το προϊόν: "{state['query']}"

Αποτελέσματα web αναζήτησης:
{web_text}

Βάσει των παραπάνω και της γενικής σου γνώσης για το προϊόν, επέστρεψε JSON με:
- name: πλήρες όνομα προϊόντος
- category: κατηγορία (π.χ. Smartphones, Laptops, Components)
- description: σύντομη περιγραφή 1-2 προτάσεις
- attributes: dict με τα βασικά τεχνικά χαρακτηριστικά
- price: τιμή αν αναφέρεται (αριθμός ή null)

Απάντησε ΜΟΝΟ με έγκυρο JSON, χωρίς markdown."""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        web_product = json.loads(content.strip())
    except Exception:
        web_product = {"name": state["query"], "category": "", "attributes": {}}

    attrs = web_product.get("attributes") or {}
    search_query = f"{web_product.get('name', '')} {web_product.get('category', '')} {' '.join(f'{k} {v}' for k, v in attrs.items())}"

    similar_from_db = search_by_text(search_query.strip(), top_k=5, threshold=0.30)

    return {**state, "web_product": web_product, "suggestions": similar_from_db}


def _web_product_is_useful(wp: dict, query: str) -> bool:
    has_category = bool(wp.get("category", "").strip())
    has_attrs = bool(wp.get("attributes"))
    has_description = bool(wp.get("description", "").strip())
    has_name = bool(wp.get("name", "").strip())
    return has_name and (has_category or has_attrs or has_description)


def node_save_web_product(state: SearchState) -> SearchState:
    wp = state.get("web_product", {})
    if not wp.get("name") or not _web_product_is_useful(wp, state["query"]):
        return {**state, "saved_to_db": False}

    try:
        add_product(
            name=wp.get("name", state["query"]),
            code=f"WEB-{uuid.uuid4().hex[:8].upper()}",
            description=wp.get("description", ""),
            category=wp.get("category", ""),
            attributes=wp.get("attributes") or {},
            price=float(wp["price"]) if wp.get("price") else None,
        )
        saved = True
    except Exception:
        saved = False

    return {**state, "saved_to_db": saved}


def node_build_similar(state: SearchState) -> SearchState:
    suggestions = list(state["db_results"])
    seen_ids = {r["id"] for r in suggestions}

    for product in state["db_results"][:2]:
        for similar in get_similar_to_product(product["id"], top_k=3):
            if similar["id"] not in seen_ids:
                seen_ids.add(similar["id"])
                suggestions.append(similar)

    return {**state, "suggestions": suggestions}


def node_answer(state: SearchState) -> SearchState:
    if state.get("used_web"):
        wp = state.get("web_product", {})

        if not _web_product_is_useful(wp, state["query"]):
            return {**state, "answer": "Το ερώτημα είναι πολύ αόριστο. Δοκίμασε να αναζητήσεις συγκεκριμένο προϊόν, π.χ. «laptop gaming 16GB RAM» ή «Poco F3»."}

        similar = [p for p in state.get("suggestions", []) if p.get("similarity", 0) >= 0.50]

        web_section = (
            f"**Προϊόν από web:** {wp.get('name', state['query'])}\n"
            f"**Κατηγορία:** {wp.get('category', '-')}\n"
            f"**Περιγραφή:** {wp.get('description', '-')}\n"
            f"**Χαρακτηριστικά:** {', '.join(f'{k}: {v}' for k, v in (wp.get('attributes') or {}).items()) or '-'}\n"
            f"**Τιμή:** {'€' + str(wp['price']) if wp.get('price') else 'Δεν αναφέρεται'}"
        )
        if similar:
            sim_list = "\n".join(
                f"- {p['name']} ({p['code']}) | {p.get('category', '')} | "
                f"{'€' + str(p['price']) if p.get('price') else 'N/A'} | "
                f"ομοιότητα: {p['similarity']:.0%}"
                for p in similar[:6]
            )
            answer = web_section + f"\n\n**Παρόμοια προϊόντα από τη βάση:**\n{sim_list}"
        else:
            answer = web_section + "\n\n*Δεν βρέθηκαν παρόμοια προϊόντα στη βάση.*"
    else:
        db_results = state.get("db_results", [])
        suggestions = state.get("suggestions", [])
        if not suggestions:
            return {**state, "answer": "Δεν βρέθηκαν αποτελέσματα."}

        found_ids = {r["id"] for r in db_results}
        direct = db_results
        similar = [p for p in suggestions if p["id"] not in found_ids]

        direct_list = "\n".join(
            f"- {p['name']} ({p['code']}) | {p.get('category', '')} | "
            f"{'€' + str(p['price']) if p.get('price') else 'N/A'}"
            for p in direct
        )
        similar_list = "\n".join(
            f"- {p['name']} ({p['code']}) | {p.get('category', '')} | "
            f"{'€' + str(p['price']) if p.get('price') else 'N/A'}"
            for p in similar[:5]
        )

        context = f"Αποτελέσματα για \"{state['query']}\":\n{direct_list}"
        if similar_list:
            context += f"\n\nΠαρόμοια προϊόντα:\n{similar_list}"

        prompt = f"""{context}

Παρουσίασε πρώτα τα άμεσα αποτελέσματα και μετά (αν υπάρχουν) τα παρόμοια. Σύντομα στα ελληνικά."""
        answer = llm.invoke([HumanMessage(content=prompt)]).content

    return {**state, "answer": answer}


def build_graph():
    g = StateGraph(SearchState)
    g.add_node("db_search", node_db_search)
    g.add_node("web_search", node_web_search)
    g.add_node("extract_and_match", node_extract_and_match)
    g.add_node("save_web_product", node_save_web_product)
    g.add_node("build_similar", node_build_similar)
    g.add_node("answer", node_answer)

    g.set_entry_point("db_search")
    g.add_conditional_edges("db_search", node_decide, {
        "suggest": "build_similar",
        "web_search": "web_search",
    })
    g.add_edge("web_search", "extract_and_match")
    g.add_edge("extract_and_match", "save_web_product")
    g.add_edge("save_web_product", "answer")
    g.add_edge("build_similar", "answer")
    g.add_edge("answer", END)

    return g.compile()


graph = build_graph()


def run_search(query: str) -> dict:
    return graph.invoke(SearchState(
        query=query,
        db_results=[],
        web_results=[],
        web_product={},
        suggestions=[],
        answer="",
        used_web=False,
        saved_to_db=False,
    ))
