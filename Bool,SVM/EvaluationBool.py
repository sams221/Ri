import os
import csv
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict

# ===================== PATHS =====================
docs_path = r"C:\Users\hp\Desktop\Projet RI\Ri\Docs"
queries_path = r"C:\Users\hp\Desktop\Projet RI\Ri\Queries"
relevances_path = r"C:\Users\hp\Desktop\Projet RI\Ri\Relevances"
output_path = r"C:\Users\hp\Desktop\Projet RI\Ri\Tp1Resultat"

models = ["BoolClassique", "Fuzzy"]

os.makedirs(output_path, exist_ok=True)
for m in models:
    os.makedirs(os.path.join(output_path, m), exist_ok=True)

# ===================== NLP SETUP =====================
import nltk
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
stemmer = PorterStemmer()

def clean_text(text):
    tokens = tokenizer.tokenize(text)
    return [stemmer.stem(w.lower()) for w in tokens if w.lower() not in stop_words]

# ===================== READ DOCUMENTS =====================
documents = {}
doc_num_map = {}
for idx, f in enumerate(sorted(os.listdir(docs_path)), start=1):
    if f.endswith(".txt"):
        doc_id = f.replace(".txt", "")
        doc_num_map[doc_id] = idx
        with open(os.path.join(docs_path, f), encoding="utf-8") as file:
            documents[doc_id] = clean_text(file.read())

# ===================== READ QUERIES =====================
queries = {}
for idx, f in enumerate(sorted(os.listdir(queries_path)), start=1):
    with open(os.path.join(queries_path, f), encoding="utf-8") as file:
        queries[idx] = clean_text(file.read())

# ===================== READ RELEVANCES =====================
relevances = {}
for f in os.listdir(relevances_path):
    if f.endswith(".txt"):
        q_num = int(os.path.splitext(f)[0])
        with open(os.path.join(relevances_path, f), encoding="utf-8") as file:
            relevances[q_num] = [int(x) for x in file.read().splitlines()]

# ===================== METRICS FUNCTIONS =====================
def precision_recall_f1(pred, true):
    pred_set, true_set = set(pred), set(true)
    tp = len(pred_set & true_set)
    precision = tp / len(pred) if pred else 0
    recall = tp / len(true) if true else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall>0 else 0
    return precision, recall, f1

def precision_at_k(pred, true, k):
    return len(set(pred[:k]) & set(true)) / k

def r_precision(pred, true):
    R = len(true)
    return len(set(pred[:R]) & set(true)) / R if R>0 else 0

def reciprocal_rank(pred, true):
    for i, doc in enumerate(pred):
        if doc in true:
            return 1/(i+1)
    return 0

def dcg(pred, true, k):
    return sum(1 / math.log2(i+2) for i, doc in enumerate(pred[:k]) if doc in true)

def ndcg(pred, true, k):
    ideal = dcg(true, true, k)
    return dcg(pred, true, k)/ideal if ideal>0 else 0

def average_precision(pred, true):
    ap = 0
    num_rel = 0
    for i, doc in enumerate(pred):
        if doc in true:
            num_rel += 1
            ap += num_rel/(i+1)
    return ap/len(true) if true else 0

def interpolated_ap(pred, true):
    recall_points = [i/10 for i in range(11)]
    prec_at_recall = []
    for r in recall_points:
        prec = max([precision_recall_f1(pred[:i+1], true)[0] for i in range(len(pred)) if (i+1)/len(true)>=r] or [0])
        prec_at_recall.append(prec)
    return sum(prec_at_recall)/len(prec_at_recall)

# ===================== RUN MODELS =====================
model_results = {}

# -------- Boolean Classique --------
bool_results = {}
for q_num, q_terms in queries.items():
    relevant_docs = []
    q_terms_set = set(q_terms)
    for doc_id, terms in documents.items():
        if q_terms_set.issubset(set(terms)):
            relevant_docs.append(doc_num_map[doc_id])
    bool_results[q_num] = sorted(relevant_docs)
    with open(os.path.join(output_path, "BoolClassique", f"{q_num}.txt"), "w") as f:
        f.write("\n".join(map(str, bool_results[q_num])))
model_results["BoolClassique"] = bool_results

# -------- Fuzzy Boolean --------
fuzzy_results = {}
for q_num, q_terms in queries.items():
    relevant_docs = []
    for doc_id, terms in documents.items():
        freq = defaultdict(int)
        for t in terms:
            freq[t] += 1
        maxf = max(freq.values()) if freq else 1
        weights = [(freq[t]/maxf) if t in freq else 0 for t in q_terms]
        score = min(weights) if weights else 0
        if score > 0:
            relevant_docs.append(doc_num_map[doc_id])
    fuzzy_results[q_num] = sorted(relevant_docs)
    with open(os.path.join(output_path, "Fuzzy", f"{q_num}.txt"), "w") as f:
        f.write("\n".join(map(str, fuzzy_results[q_num])))
model_results["Fuzzy"] = fuzzy_results

# ===================== EVALUATION =====================
for model, results in model_results.items():
    # CSV per query
    csv_file = os.path.join(output_path, f"{model}_per_query.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Query_ID","Precision","Recall","F1","Precision@5","Precision@10",
            "R-Precision","Reciprocal Rank","DCG@20","nDCG@20","Average Precision","Interpolated AP"
        ])
        all_ap, all_interpolated_ap = [], []
        for q_num, rel_docs in relevances.items():
            pred_docs = results.get(q_num, [])
            precision, recall, f1 = precision_recall_f1(pred_docs, rel_docs)
            ap = average_precision(pred_docs, rel_docs)
            iap = interpolated_ap(pred_docs, rel_docs)
            all_ap.append(ap)
            all_interpolated_ap.append(iap)
            p5 = precision_at_k(pred_docs, rel_docs, 5)
            p10 = precision_at_k(pred_docs, rel_docs, 10)
            rprec = r_precision(pred_docs, rel_docs)
            rr = reciprocal_rank(pred_docs, rel_docs)
            dcg20 = dcg(pred_docs, rel_docs, 20)
            ndcg20 = ndcg(pred_docs, rel_docs, 20)

            writer.writerow([q_num, precision, recall, f1, p5, p10, rprec, rr, dcg20, ndcg20, ap, iap])

    # CSV metrics
    csv_metrics = os.path.join(output_path, f"{model}_metrics.csv")
    MAP = sum(all_ap)/len(all_ap) if all_ap else 0
    Interpolated_MAP = sum(all_interpolated_ap)/len(all_interpolated_ap) if all_interpolated_ap else 0
    gain_percent = sum([ndcg(results.get(q, []), relevances[q], 20) for q in sorted(relevances.keys())[:10]])/10*100
    with open(csv_metrics, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric","Value"])
        writer.writerow(["MAP", MAP])
        writer.writerow(["Interpolated MAP", Interpolated_MAP])
        writer.writerow(["Gain (%)", gain_percent])
