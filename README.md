# Product Search System

## 1. Περιγραφή

Σύστημα αναζήτησης προϊόντων που χρησιμοποιεί RAG για να βρει αποτελέσματα από τοπική βάση δεδομένων. Αν δεν βρεθεί κάτι, κάνει αυτόματα αναζήτηση στο διαδίκτυο, εξάγει τα χαρακτηριστικά του προϊόντος και επιστρέφει παρόμοια προϊόντα από τη βάση. Το προϊόν που βρέθηκε online αποθηκεύεται αυτόματα ώστε η επόμενη αναζήτηση να το βρει τοπικά.

Το backend είναι FastAPI και το UI Gradio. Το Gradio καλεί το FastAPI μέσω HTTP, οπότε το API λειτουργεί και ανεξάρτητα.

## 2. Installation

Πρώτα χρειάζεται PostgreSQL με pgvector μέσω Docker:

```bash
docker run -d --name pgvector -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg17
```

Μετά δημιουργία της βάσης και εγκατάσταση του pgvector extension:

```bash
docker exec pgvector psql -U postgres -c "CREATE DATABASE products_db;"
docker exec pgvector psql -U postgres -d products_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Μετά εγκατάσταση των dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Αντιγραφή του `.env.example` σε `.env` και συμπλήρωση των μεταβλητών:

```
OPENAI_API_KEY=...
TAVILY_API_KEY=...
POSTGRES_PASSWORD=postgres
```

Φόρτωση των προϊόντων από το CSV στη βάση (τρέχει μία φορά):

```bash
python seed_products.py
```

## 3. Run backend

```bash
python app.py
```

Ο server ξεκινά στο `http://localhost:8000`.

## 4. Run UI

Το UI φορτώνει αυτόματα μαζί με τον server στο `http://localhost:8000/ui`.

## 5. API Endpoints

| Method | URL | Περιγραφή |
|--------|-----|-----------|
| POST | `/search` | Αναζήτηση προϊόντος |
| GET | `/ui` | Gradio interface |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc documentation |

Παράδειγμα request στο `/search`:

```json
POST /search
{
  "query": "laptop gaming 16GB"
}
```

## 6. GenAI Logic

Η αναζήτηση γίνεται με LangGraph agent που ακολουθεί αυτή τη ροή:

1. Το query γίνεται embedding μέσω OpenAI και συγκρίνεται με τα embeddings της βάσης (pgvector)
2. Αν βρεθούν αποτελέσματα, επιστρέφονται μαζί με παρόμοια προϊόντα βάσει vector similarity
3. Αν δεν βρεθεί τίποτα, γίνεται αναζήτηση στο διαδίκτυο μέσω Tavily
4. Το LLM εξάγει τα χαρακτηριστικά του προϊόντος από τα web αποτελέσματα
5. Με βάση αυτά τα χαρακτηριστικά γίνεται νέα vector αναζήτηση στη βάση για παρόμοια
6. Το προϊόν αποθηκεύεται αυτόματα στη βάση για μελλοντικές αναζητήσεις
7. Το LLM συνθέτει την τελική απάντηση βάσει μόνο των ανακτημένων δεδομένων
