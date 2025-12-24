import utils
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import pandas as pd


queries = utils.load_queries()

tokenizer = RegexpTokenizer(
    r'(?:[A-Za-z]\.)+'
    r'|[A-Za-z]+[\-@]\d+(?:\.\d+)?'
    r'|\d+(?:[\.\,\-]\d+)*%?'
    r'|[A-Za-z]+'
)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def query_cleaning(query):
  # Tokenize and clean query
  tokens = tokenizer.tokenize(query)
  clean_query = [word.lower() for word in tokens if word.lower() not in stop_words]
  
  Porter = PorterStemmer()
  TermsPorter = [Porter.stem(t) for t in clean_query]
  
  return TermsPorter

def calc_weight_Word_query(query, word):
  if word in query:
    return 1
  return 0

clean_queries =[]
for q in queries:
    clean_queries.append(query_cleaning(q))
    
data = pd.read_csv(
    "Docterm_matrix.txt",
    sep=r"\s+", header=None,
    names=["Document", "Term", "Frequency", "Weight"]
)

# Remove the first row (index 0) and reset index
data = data.iloc[1:].reset_index(drop=True)

print("📄 Aperçu du DataFrame original :")
print(data.head())

df_agg = data.groupby(["Document", "Term"], as_index=False)["Weight"]

print(df_agg.head())