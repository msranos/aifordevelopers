import os
import json
import csv

from db import init_db
from vector_store import add_product

CSV_PATH = os.path.join(os.path.dirname(__file__), "products.csv")


def load_from_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                attributes = json.loads(row.get("attributes") or "{}")
            except json.JSONDecodeError:
                attributes = {}
            price_raw = row.get("price", "").strip()
            price = float(price_raw) if price_raw else None
            add_product(
                name=row["name"].strip(),
                code=row["code"].strip(),
                description=row.get("description", "").strip(),
                category=row.get("category", "").strip(),
                attributes=attributes,
                price=price,
            )
            print(f"  ✓ {row['name']}")


if __name__ == "__main__":
    init_db()
    load_from_csv(CSV_PATH)
    print("Done.")
