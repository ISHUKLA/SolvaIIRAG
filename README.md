# SolvaIIRAG

Application Streamlit pour interroger un corpus Solvabilite II avec un moteur RAG.

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app_solvency_rag_llm.py
```

## Deployer sur Streamlit Community Cloud

1. Publier ce dossier dans un repository GitHub.
2. Creer une app sur https://share.streamlit.io.
3. Selectionner le repository, la branche `main`, puis le fichier d'entree `app_solvency_rag_llm.py`.
4. Ajouter `GROQ_API_KEY` dans les secrets Streamlit si la synthese LLM doit etre activee.

Sans cle Groq, l'application reste utilisable en mode recherche/extraits sources.

## Secrets

Ne pas committer `.streamlit/secrets.toml` ni de fichier `.env`.
