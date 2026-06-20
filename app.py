"""FastAPI + Gradio product search system."""

import httpx
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from db import init_db
from agent import run_search

# ── FastAPI ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Product Search System")


@app.on_event("startup")
def startup():
    init_db()


class SearchRequest(BaseModel):
    query: str


class ProductResult(BaseModel):
    name: str
    code: str
    category: str
    price: float | None
    similarity: float | None


class SearchResponse(BaseModel):
    query: str
    answer: str
    used_web: bool
    saved_to_db: bool
    results: list[ProductResult]
    web_product: dict | None


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    result = run_search(req.query)
    products = result.get("db_results") or result.get("suggestions") or []
    results = [
        ProductResult(
            name=p.get("name", ""),
            code=p.get("code", ""),
            category=p.get("category", ""),
            price=float(p["price"]) if p.get("price") else None,
            similarity=round(float(p["similarity"]), 4) if p.get("similarity") else None,
        )
        for p in products
    ]
    return SearchResponse(
        query=req.query,
        answer=result.get("answer", ""),
        used_web=result.get("used_web", False),
        saved_to_db=result.get("saved_to_db", False),
        results=results,
        web_product=result.get("web_product") or None,
    )


# ── Gradio (καλεί το FastAPI /search) ────────────────────────────────────────

def gradio_search(query: str):
    if not query.strip():
        return "Παρακαλώ εισάγετε αναζήτηση.", None

    with httpx.Client() as client:
        response = client.post(
            "http://localhost:8000/search",
            json={"query": query},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

    source = "🌐 Web search + Βάση" if data["used_web"] else "🗄️ Βάση δεδομένων"
    saved_note = "\n\n> ✅ Το προϊόν αποθηκεύτηκε αυτόματα στη βάση." if data.get("saved_to_db") else ""
    answer = f"**{source}**{saved_note}\n\n{data['answer']}"

    # Web product card
    wp = data.get("web_product")
    if data["used_web"] and wp and wp.get("name"):
        attrs = ", ".join(f"{k}: {v}" for k, v in (wp.get("attributes") or {}).items())
        web_card = (
            f"### Προϊόν από web: {wp['name']}\n"
            f"- **Κατηγορία:** {wp.get('category', '-')}\n"
            f"- **Περιγραφή:** {wp.get('description', '-')}\n"
            f"- **Χαρακτηριστικά:** {attrs or '-'}\n"
            f"- **Τιμή:** {'€' + str(wp['price']) if wp.get('price') else 'Δεν αναφέρεται'}"
        )
        answer = web_card + "\n\n---\n\n" + answer

    # Results table
    import pandas as pd
    products = data.get("results", [])
    if products:
        df = pd.DataFrame([{
            "Όνομα": p["name"],
            "Κωδικός": p["code"],
            "Κατηγορία": p["category"],
            "Τιμή": f"€{p['price']:.2f}" if p.get("price") else "-",
            "Ομοιότητα": f"{p['similarity']*100:.0f}%" if p.get("similarity") else "-",
        } for p in products])
    else:
        df = pd.DataFrame()

    return answer, gr.Dataframe(value=df, visible=bool(products))


with gr.Blocks(title="Product Search", theme=gr.themes.Soft()) as gradio_app:
    gr.Markdown("# 🔍 Σύστημα Αναζήτησης Προϊόντων")
    gr.Markdown("Αναζήτηση με RAG — αν δεν βρεθεί τοπικά, ψάχνει στο web.")

    with gr.Row():
        query_input = gr.Textbox(
            label="Αναζήτηση",
            placeholder="π.χ. laptop gaming 16GB, ασύρματα ακουστικά ANC...",
            scale=4,
        )
        search_btn = gr.Button("🔍 Αναζήτηση", variant="primary", scale=1)

    answer_out = gr.Markdown()
    results_table = gr.Dataframe(label="Αποτελέσματα", interactive=False, visible=False)

    search_btn.click(gradio_search, inputs=[query_input], outputs=[answer_out, results_table])
    query_input.submit(gradio_search, inputs=[query_input], outputs=[answer_out, results_table])



# ── Mount Gradio στο FastAPI ──────────────────────────────────────────────────

app = gr.mount_gradio_app(app, gradio_app, path="/ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
