import os
import re

def split_medline_collection(input_file, output_dir):
    # Création du dossier de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Utilisation d'une expression régulière pour capturer l'ID et le contenu après .W
    # Le flag re.DOTALL permet au '.' de correspondre aussi aux sauts de ligne
    pattern = re.compile(r'\.I\s+(\d+)\n\.W\n(.*?)(?=\n\.I|\Z)', re.DOTALL)
    
    matches = pattern.findall(content)

    for doc_id, doc_text in matches:
        file_path = os.path.join(output_dir, f"query_{doc_id}.txt")
        with open(file_path, 'w', encoding='utf-8') as out_f:
            # Nettoyage des espaces blancs inutiles au début et à la fin
            out_f.write(doc_text.strip())

    print(f"Traitement terminé : {len(matches)} documents créés dans '{output_dir}'.")

# Utilisation du script
split_medline_collection('C:\\Users\\hp\\Desktop\\TPs\\Ri5\\Data\\MED.QRY', 'C:\\Users\\hp\\Desktop\\TPs\\Ri5\\Queries')