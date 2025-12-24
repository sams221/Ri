import os
import csv
import math
import nltk
import numpy as np
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ===================== PATHS =====================
DOCS_DIR = r"C:\Users\hp\Desktop\Projet RI\Ri\Docs"
QUERIES_DIR = r"C:\Users\hp\Desktop\Projet RI\Ri\Queries"
INDEX_FILE = r"C:\Users\hp\Desktop\Projet RI\Ri\invertedIndex.txt"
REL_DIR = r"C:\Users\hp\Desktop\Projet RI\Ri\Relevances"
OUT_DIR = r"C:\Users\hp\Desktop\Projet RI\Ri\Tp1Resultat\SVM"

os.makedirs(OUT_DIR, exist_ok=True)

# ===================== NLP =====================
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
tokenizer = RegexpTokenizer(r'(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+(?:[\.\,\-]\d+)*%?|[A-Za-z]+')
stemmer = PorterStemmer()

def clean_text(text):
    tokens = tokenizer.tokenize(text)
    return [stemmer.stem(t.lower()) for t in tokens if t.lower() not in stop_words]

# ===================== READ DOCUMENT IDS =====================
doc_num = {}
for i, f in enumerate(sorted(os.listdir(DOCS_DIR)), start=1):
    if f.endswith(".txt"):
        doc_num[f.replace(".txt", "")] = i

# ===================== READ QUERIES =====================
queries = {}
for i, f in enumerate(sorted(os.listdir(QUERIES_DIR)), start=1):
    with open(os.path.join(QUERIES_DIR, f), encoding="utf-8") as file:
        queries[i] = clean_text(file.read())

# ===================== READ INVERTED INDEX =====================
docs_vectors = defaultdict(dict)
with open(INDEX_FILE, encoding="utf-8") as f:
    for line in f:
        term, doc, freq, weight = line.split()
        docs_vectors[doc][term] = float(weight)

# ===================== READ RELEVANCES =====================
relevances = {}
for f in os.listdir(REL_DIR):
    if f.endswith(".txt"):
        qid = int(f.replace(".txt", ""))
        with open(os.path.join(REL_DIR, f)) as file:
            relevances[qid] = [int(x.strip()) for x in file.readlines()]

# ===================== COSINE SIM =====================
def cosine(qv, dv):
    dot = sum(qv[t] * dv.get(t, 0) for t in qv)
    nq = math.sqrt(sum(v*v for v in qv.values()))
    nd = math.sqrt(sum(v*v for v in dv.values()))
    return dot / (nq * nd) if nq and nd else 0

# ===================== METRICS =====================
def precision_recall_f1(pred, true):
    tp = len(set(pred) & set(true))
    p = tp / len(pred) if pred else 0
    r = tp / len(true) if true else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    return p, r, f1

def precision_at_k(pred, true, k):
    return len(set(pred[:k]) & set(true)) / k if k else 0

def r_precision(pred, true):
    R = len(true)
    return len(set(pred[:R]) & set(true)) / R if R else 0

def reciprocal_rank(pred, true):
    for i, d in enumerate(pred):
        if d in true:
            return 1/(i+1)
    return 0

def dcg(pred, true, k):
    return sum(1/math.log2(i+2) for i,d in enumerate(pred[:k]) if d in true)

def ndcg(pred, true, k):
    ideal = dcg(true, true, k)
    return dcg(pred, true, k) / ideal if ideal else 0

def average_precision(pred, true):
    score, hits = 0, 0
    for i, d in enumerate(pred):
        if d in true:
            hits += 1
            score += hits / (i+1)
    return score / len(true) if true else 0

def interpolated_ap(pred, true):
    precisions, recalls = [], []
    for i in range(len(pred)):
        p, r, _ = precision_recall_f1(pred[:i+1], true)
        precisions.append(p)
        recalls.append(r)

    interp = []
    for r in [i/10 for i in range(11)]:
        interp.append(max([p for p,rc in zip(precisions,recalls) if rc >= r] or [0]))
    return sum(interp)/11

# ===================== RUN SVM =====================
results = {}

for qid, qterms in queries.items():
    qv = {t: qterms.count(t) for t in set(qterms)}
    scores = []

    for doc, dv in docs_vectors.items():
        score = cosine(qv, dv)
        scores.append((doc_num[doc], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_docs = [d for d,_ in scores]
    results[qid] = ranked_docs

    with open(os.path.join(OUT_DIR, f"{qid}.txt"), "w") as f:
        f.write("\n".join(map(str, ranked_docs)))

# ===================== EVALUATION =====================
csv_per_query = os.path.join(OUT_DIR, "SVM_per_query.csv")
metrics = [
    "Precision","Recall","F1","P@5","P@10",
    "R-Precision","RR","DCG@20","nDCG@20","AP","Interpolated AP"
]

acc = {m: [] for m in metrics}

with open(csv_per_query, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Query_ID"] + metrics)

    for q in sorted(relevances):
        pred, true = results[q], relevances[q]

        p,r,f1 = precision_recall_f1(pred,true)
        vals = [
            p, r, f1,
            precision_at_k(pred,true,5),
            precision_at_k(pred,true,10),
            r_precision(pred,true),
            reciprocal_rank(pred,true),
            dcg(pred,true,20),
            ndcg(pred,true,20),
            average_precision(pred,true),
            interpolated_ap(pred,true)
        ]

        for m,v in zip(metrics, vals):
            acc[m].append(v)

        writer.writerow([q] + [round(v,4) for v in vals])

    writer.writerow(["MEAN"] + [round(np.mean(acc[m]),4) for m in metrics])

# ===================== GLOBAL METRICS =====================
csv_global = os.path.join(OUT_DIR, "SVM_metrics.csv")

MAP = np.mean([average_precision(results[q],relevances[q]) for q in relevances])
IMAP = np.mean([interpolated_ap(results[q],relevances[q]) for q in relevances])
GAIN = np.mean([ndcg(results[q],relevances[q],20) for q in list(relevances)[:10]])*100

with open(csv_global,"w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric","Value"])
    writer.writerow(["MAP",round(MAP,4)])
    writer.writerow(["Interpolated MAP",round(IMAP,4)])
    writer.writerow(["Gain (%)",round(GAIN,4)])

print(" Évaluation terminée avec succès.")
