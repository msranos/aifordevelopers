import json
from sqlalchemy import text
from db import engine
from embeddings import embed_text, product_to_text


def _vec_str(emb) -> str:
    return '[' + ','.join(map(str, emb)) + ']'


def add_product(name: str, code: str, description: str = "", category: str = "",
                attributes: dict = None, price: float = None):
    attributes = attributes or {}
    emb = embed_text(product_to_text(name, code, description, attributes))
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO products (name, code, description, category, attributes, price, embedding)
            VALUES (:name, :code, :desc, :cat, CAST(:attrs AS jsonb), :price, CAST(:emb AS vector))
            ON CONFLICT (code) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                category = EXCLUDED.category,
                attributes = EXCLUDED.attributes,
                price = EXCLUDED.price,
                embedding = EXCLUDED.embedding
        """), {
            "name": name, "code": code, "desc": description,
            "cat": category, "attrs": json.dumps(attributes),
            "price": price, "emb": _vec_str(emb)
        })
        conn.commit()


def search_by_text(query: str, top_k: int = 5, threshold: float = 0.4) -> list[dict]:
    emb = embed_text(query)
    vec = _vec_str(emb)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, name, code, description, category, attributes, price,
                   1 - (embedding <=> CAST(:emb AS vector)) AS similarity
            FROM products
            WHERE 1 - (embedding <=> CAST(:emb AS vector)) > :threshold
            ORDER BY similarity DESC
            LIMIT :k
        """), {"emb": vec, "threshold": threshold, "k": top_k})
        return [dict(r._mapping) for r in rows]


def get_similar_to_product(product_id: int, top_k: int = 5) -> list[dict]:
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT embedding::text FROM products WHERE id = :id"
        ), {"id": product_id}).fetchone()
        if not row:
            return []
        rows = conn.execute(text("""
            SELECT id, name, code, description, category, attributes, price,
                   1 - (embedding <=> CAST(:emb AS vector)) AS similarity
            FROM products
            WHERE id != :id
            ORDER BY similarity DESC
            LIMIT :k
        """), {"emb": row[0], "id": product_id, "k": top_k})
        return [dict(r._mapping) for r in rows]
