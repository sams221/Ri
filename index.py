import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.stem import LancasterStemmer
import collections
import math


collection_path = r"C:\Users\hp\Desktop\TPs\Ri5\Docs"
porter_docs = {}
documents = {}

term_doc_freq = collections.defaultdict(set)

for filename in os.listdir(collection_path):
    if filename.endswith(".txt"):
        doc_id = filename.split('.')[0]

        with open(os.path.join(collection_path, filename), 'r', encoding='utf-8') as file:
                documents[doc_id] = file.read()
        
        ExpReg = RegexpTokenizer(r'(?:[A-Za-z]\.)+'
                                     r'|[A-Za-z]+[\-@]\d+(?:\.\d+)?'
                                     r'|\d+(?:[\.\,\-]\d+)*%?'
                                     r'|[A-Za-z]+'
                                    )
        stopWords = stopwords.words('english')
        terms = ExpReg.tokenize(documents[doc_id])
        TermsNotStop = [t for t in terms if t.lower() not in stopWords]
        #porter stemming
        Porter = PorterStemmer()
        TermsPorter = [Porter.stem(t) for t in TermsNotStop]
        porter_docs[doc_id] = TermsPorter
        #Update document frequencies for Porter
        for term in set(TermsPorter):
            term_doc_freq[term].add(doc_id)
            
           
        
N=len(documents)
print(f"Total number of documents: {N}")

def compute_tfidf_weights(dict_docs, term_doc_freq, N):
    
    doc_weights = {}
    term_weights = collections.defaultdict(dict)
    
    for doc_id, terms in dict_docs.items():
        term_freq = collections.Counter(terms)
        max_freq = max(term_freq.values()) if term_freq else 1
        
        doc_weights[doc_id] = {}
        
        for term, freq in term_freq.items():
            tf = freq / max_freq
            n_t = len(term_doc_freq.get(term, set()))
            idf = math.log10((N / n_t) + 1)
          
            weight = tf * idf
            
            doc_weights[doc_id][term] = weight
            term_weights[term][doc_id] = weight
    
    return doc_weights, term_weights

# Compute weights for both stemmers
porter_doc_weights, porter_term_weights = compute_tfidf_weights(porter_docs, term_doc_freq, N)

porter_descriptor_file = open("C:\\Users\\hp\\Desktop\\TPs\\Ri5\\DocTerm_matrix.txt", 'w', encoding='utf-8')

porter_descriptor_file.write("<Document> <Term> <Freq> <Weight>\n")
porter_count = 0
for doc_id, terms in sorted(porter_docs.items()):
    term_freq = collections.Counter(terms)
    for term, freq in sorted(term_freq.items()):
        weight = porter_doc_weights[doc_id].get(term, 0)
        porter_descriptor_file.write(f"{doc_id} {term} {freq} {weight:.6f}\n")
        porter_count += 1
    
porter_inverted_file = open("C:\\Users\\hp\\Desktop\\TPs\\Ri5\\InvertedIndex.txt", 'w', encoding='utf-8')

    # Write Porter inverted index file
porter_inverted_file.write("<Term> <Document> <Freq> <Weight>\n")
porter_inv_count = 0
for term, doc_weights in sorted(porter_term_weights.items()):
    for doc_id, weight in sorted(doc_weights.items()):
        freq = porter_docs[doc_id].count(term)
        porter_inverted_file.write(f"{term} {doc_id} {freq} {weight:.6f}\n")
        porter_inv_count += 1
    # Close all files
porter_descriptor_file.close()
   
porter_inverted_file.close()
