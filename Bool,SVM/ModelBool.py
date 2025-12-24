import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict

# ================= SETUP =================
docs_path = r"C:\Users\hp\Desktop\Projet RI\Ri\Docs"
queries_path = r"C:\Users\hp\Desktop\Projet RI\Ri\Queries"
output_base = r"C:\Users\hp\Desktop\Projet RI\Ri\Tp1Resultat"

# Créer les dossiers pour BoolClassique et Fuzzy
bool_dir = os.path.join(output_base, "BoolClassique")
fuzzy_dir = os.path.join(output_base, "Fuzzy")
for d in [bool_dir, fuzzy_dir]:
    os.makedirs(d, exist_ok=True)

# ================= NLP SETUP =================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+(?:[\.\,\-]\d+)*%?|[A-Za-z]+')
stemmer = PorterStemmer()

def clean_text(text):
    tokens = tokenizer.tokenize(text)
    return [stemmer.stem(w.lower()) for w in tokens if w.lower() not in stop_words]

# ================= READ DOCUMENTS =================
documents = {}
doc_num_map = {}
for idx, f in enumerate(sorted(os.listdir(docs_path)), start=1):
    if f.endswith(".txt"):
        doc_id = f.replace(".txt", "")
        doc_num_map[doc_id] = idx
        with open(os.path.join(docs_path, f), encoding="utf-8") as file:
            documents[doc_id] = clean_text(file.read())

# ================= READ QUERIES =================
queries = {}
for idx, f in enumerate(sorted(os.listdir(queries_path)), start=1):
    with open(os.path.join(queries_path, f), encoding="utf-8") as file:
        queries[idx] = clean_text(file.read())

# ================= BOOLEAN MODEL (AND implicite) =================
for q_num, q_terms in queries.items():
    q_terms_set = set(q_terms)
    relevant_docs = [
        doc_num_map[doc] for doc in documents 
        if q_terms_set.issubset(set(documents[doc]))
    ]
    with open(os.path.join(bool_dir, f"{q_num}.txt"), "w") as f:
        for doc_id in sorted(relevant_docs):
            f.write(f"{doc_id}\n")

# ================= FUZZY BOOLEAN MODEL =================
for q_num, q_terms in queries.items():
    relevant_docs = []
    for doc, terms in documents.items():
        freq = defaultdict(int)
        for t in terms:
            freq[t] += 1
        maxf = max(freq.values()) if freq else 1
        weights = [(freq[t]/maxf) if t in freq else 0 for t in q_terms]
        score = min(weights) if weights else 0
        if score > 0:
            relevant_docs.append(doc_num_map[doc])
    with open(os.path.join(fuzzy_dir, f"{q_num}.txt"), "w") as f:
        for doc_id in sorted(relevant_docs):
            f.write(f"{doc_id}\n")
