import os

def process_med_rel(input_file, output_dir):
    # 1. Création du dossier de destination
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Dossier '{output_dir}' créé.")

    # dictionnaire pour stocker {id_requete: [liste_docs]}
    qrels = {}

    # 2. Lecture du fichier MED.REL
    try:
        with open(input_file, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    query_id = parts[0]
                    doc_id = parts[2]
                    
                    if query_id not in qrels:
                        qrels[query_id] = []
                    qrels[query_id].append(doc_id)
        
        # 3. Écriture des fichiers individuels
        for q_id, docs in qrels.items():
            file_path = os.path.join(output_dir, f"{q_id}.txt")
            with open(file_path, 'w') as out_f:
                # On écrit chaque ID de document sur une nouvelle ligne
                out_f.write("\n".join(docs))
        
        print(f"Succès : {len(qrels)} fichiers de pertinence créés dans '{output_dir}'.")

    except FileNotFoundError:
        print(f"Erreur : Le fichier '{input_file}' est introuvable.")

# --- Exécution ---
# Assurez-vous que le fichier MED.REL est dans le même dossier que ce script
process_med_rel('C:\\Users\\hp\\Desktop\\TPs\\Ri5\\Data\\MED.REL', 'C:\\Users\\hp\\Desktop\\TPs\\Ri5\\Relevances')