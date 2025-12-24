import re
import os


def load_queries(path_query="./Queries/"):

  def natural_key(text):
      return [int(t) if t.isdigit() else t.lower()
              for t in re.split(r'(\d+)', text)]

  queries = sorted(
      (f for f in os.listdir(path_query)
      if os.path.isfile(os.path.join(path_query, f))),
      key=natural_key
  )

  queries_list = []
  for query in queries:
      with open(os.path.join(path_query, query), 'r', encoding='utf-8') as file:
          queries_list.append(file.read())
  return queries_list
        
def load_relevance():
  relevance = {}
  relevance_folder = "./Relevances/"  # adjust path

  for filename in os.listdir(relevance_folder):
      if filename.endswith(".txt"):
          query_name = filename.replace(".txt", "")  # e.g., "Query1"
          query_name = f"Query {query_name}"
          file_path = os.path.join(relevance_folder, filename)
          with open(file_path, "r") as f:
              docs = [f"doc_{line.strip()}" for line in f if line.strip()]
          relevance[query_name] = docs
          
  return relevance