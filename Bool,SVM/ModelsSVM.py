import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from collections import defaultdict
import math

# ===================== INIT =====================
Porter = PorterStemmer()
ExpReg = RegexpTokenizer(r'(?:[A-Za-z]\.)+'r'|[A-Za-z]+[\-@]\d+(?:\.\d+)?'r'|\d+(?:[\.\,\-]\d+)*%?'r'|[A-Za-z]+')

# ===================== FONCTIONS =====================

def load_inverted_index(filename):
    """Charge l'index inversé : <Term> <Document> <Freq> <Weight>"""
    inverted_index = defaultdict(dict)
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                term = parts[0].lower()
                doc_id = parts[1]
                freq = int(parts[2])
                weight = float(parts[3])
                inverted_index[term][doc_id] = {'freq': freq, 'weight': weight}
    return inverted_index

def preprocess_query(query):
    """Tokenisation et stemming"""
    terms = ExpReg.tokenize(query.lower())
    return [Porter.stem(t) for t in terms]

# ===================== MODELE VECTORIEL =====================

def cosine_similarity(query_vector, full_doc_vector, doc_vector):
    """Similarité cosinus"""
    dot_product = 0.0
    norm_query = 0.0
    norm_doc = 0.0
    
    for term in query_vector:
        dot_product += query_vector[term] * doc_vector.get(term, 0.0)
        norm_query += query_vector[term] ** 2
    
    for term in full_doc_vector:
        norm_doc += full_doc_vector[term] ** 2
    
    if norm_query == 0 or norm_doc == 0:
        return 0.0
    
    return dot_product / (math.sqrt(norm_query) * math.sqrt(norm_doc))

def vector_space_model_cosine(query, inverted_index):
    """Calcul des documents pertinents pour une requête avec cosine similarity"""
    query_terms = preprocess_query(query)
    query_vector = {term: 1.0 for term in query_terms}
    
    # Tous les documents de l'index
    all_docs = set()
    for term_docs in inverted_index.values():
        all_docs.update(term_docs.keys())
    
    doc_scores = {}
    for doc_id in all_docs:
        doc_vector = {}
        full_doc_vector = {}
        for term, doc_data in inverted_index.items():
            full_doc_vector[term] = doc_data.get(doc_id, {}).get('weight', 0.0)
        for term in query_terms:
            doc_vector[term] = inverted_index.get(term, {}).get(doc_id, {}).get('weight', 0.0)
        
        score = cosine_similarity(query_vector, full_doc_vector, doc_vector)
        doc_scores[doc_id] = score
    
    # Retourner documents triés par score décroissant (même 0)
    return [doc_id for doc_id, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)]

# ===================== MAIN =====================

def main():
    inverted_index_file = r"C:\Users\hp\Desktop\Projet RI\Ri\invertedIndex.txt"
    queries_folder = r"C:\Users\hp\Desktop\Projet RI\Ri\Queries"
    result_folder = r"C:\Users\hp\Desktop\Projet RI\Ri\Resultat\SVM"
    
    os.makedirs(result_folder, exist_ok=True)
    
    print("Chargement de l'index inversé...")
    inverted_index = load_inverted_index(inverted_index_file)
    
    query_files = [f for f in os.listdir(queries_folder) if os.path.isfile(os.path.join(queries_folder, f))]
    query_files.sort()
    
    for i, qfile in enumerate(query_files, 1):
        query_path = os.path.join(queries_folder, qfile)
        with open(query_path, 'r', encoding='utf-8') as f:
            query_text = f.read().strip()
        
        relevant_docs = vector_space_model_cosine(query_text, inverted_index)
        
        output_file = os.path.join(result_folder, f"{i}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc_id in relevant_docs:
                # Extraire uniquement le numéro après "doc_"
                if '_' in doc_id:
                    number = doc_id.split('_')[1]
                else:
                    number = doc_id
                f.write(f"{number}\n")

        print(f"Requête {i} ({qfile}) traitée. Résultats enregistrés dans {output_file}")

if __name__ == "__main__":
    main()
