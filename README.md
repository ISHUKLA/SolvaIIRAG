# SolvaIIRAG

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://solva2rag-beta.streamlit.app/)

**A deployment-ready Solvency II regulatory intelligence assistant with cited answers, embedded source documents, and a public Streamlit interface.**

SolvaIIRAG is a Retrieval-Augmented Generation (RAG) application built for insurance regulation. It helps users ask practical questions about Solvency II and inspect the source passages behind each answer. The goal is not to hide regulatory complexity behind a chatbot, but to make dense supervisory material searchable, explainable, and auditable.

The project is designed as a public portfolio application: a visitor can open the app, choose a suggested question, and receive sourced results without configuring a local folder, uploading files, or providing an API key.

![SolvaIIRAG app preview](assets/app-preview.png)

## Executive Summary

Solvency II documentation is spread across directives, delegated regulations, EIOPA material, local supervisory notices, and Q&A pages. For risk, actuarial, compliance, and consulting teams, the challenge is rarely a lack of information. The challenge is finding the relevant paragraph quickly, understanding its context, and keeping a traceable link to the source.

SolvaIIRAG addresses that workflow with:

- **A curated embedded corpus** of Solvency II documents included directly in the repository.
- **Automatic startup indexing** so the public demo works immediately for a first-time visitor.
- **Retrieval-first answers** that expose source passages instead of relying on ungrounded generation.
- **Optional LLM synthesis** for clearer explanations when a Groq API key is available.
- **A Streamlit interface** designed for a recruiter, Chief Risk Officer, or technical reviewer to understand the value in under a minute.

## What This Demonstrates

This project is meant to show more than a working chatbot. It demonstrates the ability to turn a regulatory problem into a usable product:

| Capability | How it appears in the project |
| --- | --- |
| Insurance domain understanding | Solvency II concepts such as SCR, Best Estimate, Risk Margin, ORSA, governance, SFCR/RSR, and EIOPA guidance are built into the demo flow. |
| RAG system design | Documents are loaded, chunked, indexed, retrieved, ranked, and displayed with citations. |
| Product judgment | The app no longer asks public users for a local path; the corpus is embedded and the index loads automatically. |
| Traceability | Answers remain connected to document names, pages, sections, and retrieved snippets. |
| Deployment awareness | The app supports Streamlit Community Cloud and works without requiring a private API key. |

## Key Features

- **Embedded regulatory corpus** in `Directive/`, including EU, EIOPA, ACPR, and BNB/NBB material.
- **Zero-configuration public demo**: no local path input, no file upload, no mandatory API key.
- **Automatic index loading** at app startup, so users can ask a question immediately.
- **BM25 retrieval** for fast keyword search on legal and regulatory wording.
- **Hybrid retrieval path** with Chroma and multilingual sentence embeddings when the vector index is available.
- **Optional reranking** with a cross-encoder to improve source ordering.
- **Optional Groq LLM synthesis** through `GROQ_API_KEY`; without it, the app still returns sourced retrieval results.
- **Citation-first user experience** with document names, pages, retrieved excerpts, and exportable question history.

## Example Questions

The app is structured around real regulatory questions a Solvency II user might ask:

- What does Article 101 say about the SCR?
- How is the Risk Margin calculated?
- What are the governance requirements under Solvency II?
- What does Article 45 say about ORSA?
- How is the Best Estimate defined?

## Architecture

```mermaid
flowchart LR
    Q["User question<br/><b>Example: Article 77 - Best Estimate?</b>"]

    subgraph ING["1. Document ingestion"]
        D["Solvency II corpus<br/>Directive 2009/138/EC<br/>Delegated Regulation 2015/35<br/>EIOPA - ACPR - BNB"]
        P["Multi-format parser<br/>PDF - DOCX - HTML<br/>OCR fallback"]
        C["Chunking + metadata<br/>article - page<br/>jurisdiction - version"]
    end

    subgraph RET["2. Hybrid retrieval"]
        B["BM25 Okapi<br/>Lexical search"]
        E["Multilingual embeddings<br/>multilingual-e5-large<br/>ChromaDB"]
        F["RRF fusion<br/>BM25 x 0.60<br/>Vector x 0.40<br/>Top-60 candidates"]
        R["Cross-encoder reranker<br/>multilingual ms-marco<br/>Top-8 chunks"]
    end

    subgraph GEN["3. Controlled answer generation"]
        H["Norm hierarchy<br/>Directive > Delegated Acts<br/>EIOPA > Local supervisor"]
        L["Optional LLM synthesis<br/>Groq - llama-3.3-70b<br/>Structured system prompt"]
        A["Audit log JSONL<br/>question - sources<br/>scores - model"]
    end

    O["Final answer<br/><b>Plain language - Regulation - Practice - Limits</b><br/>Citations: Article - Source - Page"]

    Q --> D --> P --> C
    C --> B
    C --> E
    B --> F
    E --> F
    F --> R
    R --> H
    H --> L
    L --> O
    R --> A
    O --> A

    classDef input fill:#F4F7FA,stroke:#4B5563,stroke-width:1.4px,color:#111827;
    classDef ingest fill:#E6F7F1,stroke:#1D9E75,stroke-width:1.6px,color:#064E3B;
    classDef lexical fill:#E8F1FC,stroke:#378ADD,stroke-width:1.6px,color:#123B66;
    classDef vector fill:#EEEAFE,stroke:#7F77DD,stroke-width:1.6px,color:#35266B;
    classDef rank fill:#FFF4DB,stroke:#EF9F27,stroke-width:1.8px,color:#7A4B00;
    classDef gen fill:#EDF8E8,stroke:#639922,stroke-width:1.6px,color:#274C0A;
    classDef audit fill:#FDEDE8,stroke:#D85A30,stroke-width:1.6px,color:#7A2411;
    classDef output fill:#EAF8F2,stroke:#15805F,stroke-width:2px,color:#064E3B;

    class Q input;
    class D,P,C ingest;
    class B lexical;
    class E vector;
    class F,R rank;
    class H,L gen;
    class A audit;
    class O output;
```

### Main Components

| File / folder | Purpose |
| --- | --- |
| `app_solvency_rag_llm.py` | Streamlit application, public UX, session state, controls, answer display, and source rendering. |
| `solvency_notebook_runtime.py` | Static Python runtime for retrieval, chunk loading, citations, fallback behavior, and answer generation. |
| `Directive/` | Embedded Solvency II corpus used by the public app. |
| `rag3_index/chunks.jsonl` | Pre-built auditable chunk store used by BM25 and retrieval fallback. |
| `RAG3_SolvencyII_improved.ipynb` | Research notebook used during experimentation and development. |
| `.streamlit/config.toml` | Streamlit deployment configuration. |

## User Journey

1. A visitor opens the public Streamlit URL.
2. The app loads the Solvency II index automatically.
3. The visitor clicks a suggested question or types a custom one.
4. The retrieval engine finds relevant regulatory passages.
5. The answer appears with citations and source excerpts.
6. If a Groq key is configured, the app adds a synthesized explanation grounded in the retrieved context.

## Technical Choices

- **BM25 first** because regulatory users often search for exact legal wording, article numbers, and defined terms.
- **Multilingual embeddings** because the corpus contains French and English supervisory material.
- **Static runtime module** to keep deployment readable and avoid dynamic notebook execution in production code.
- **Embedded corpus** so the app is usable by external reviewers without access to the developer's local machine.
- **Graceful LLM fallback** so the product still works as a cited search assistant without paid API access.

## Retrieval Evaluation Summary

The retrieval component was evaluated on **127 Solvency II questions** covering Pillar 1, Pillar 2, Pillar 3, group supervision, investments, reinsurance, internal models, prudential supervision and transversal topics.

For each question, the retriever returned the top-4 most relevant chunks. A retrieval was counted as successful when at least 50% of the expected reference terms were found in one of the retrieved chunks.

### Global Results

| Metric | Value |
|---|---:|
| Questions evaluated | 127 |
| Hit@4 | 0.890 |
| Mean MRR | 0.829 |
| Mean first hit rank | 1.19 |
| Mean term coverage | 0.682 |
| Failed questions | 14 |

### Strongest Categories

| Category | Questions | Hit rate | Mean MRR | Mean first hit rank | Mean coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pilier 1 - Fonds propres | 7 | 1.0 | 1.0 | 1.0 | 0.762 |
| Proportionnalite | 3 | 1.0 | 1.0 | 1.0 | 0.667 |
| Reassurance | 5 | 1.0 | 1.0 | 1.0 | 0.800 |

### Weakest Categories

| Category | Questions | Hit rate | Mean MRR | Mean first hit rank | Mean coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transversal | 6 | 0.667 | 0.583 | 1.25 | 0.417 |
| Risques specifiques | 6 | 0.667 | 0.667 | 1.0 | 0.583 |
| Pilier 2 - ORSA | 4 | 0.750 | 0.750 | 1.0 | 0.375 |

### Interpretation

The retrieval pipeline achieved a Hit@4 of **89.0%**, meaning that the expected source content was retrieved within the top-4 chunks for most evaluation questions. The mean first hit rank of **1.19** indicates that successful matches are generally ranked very high, often in the first retrieved positions.

Remaining failures are useful for improving chunking, query formulation, synonym handling, and coverage of regulatory terminology.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app_solvency_rag_llm.py
```

Expected repository layout:

```text
SolvaIIRAG/
├── app_solvency_rag_llm.py
├── solvency_notebook_runtime.py
├── Directive/
├── rag3_index/
├── assets/
└── requirements.txt
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new app on Streamlit Community Cloud.
3. Select branch `main` and entrypoint `app_solvency_rag_llm.py`.
4. Optional: add `GROQ_API_KEY` in Streamlit secrets to activate LLM synthesis.

The app remains usable without a Groq key in retrieval-only mode.

## Limitations and Next Steps

- This is a regulatory research assistant, not legal advice.
- The quality of answers depends on the completeness and freshness of the embedded corpus.
- LLM-generated synthesis should be reviewed against the displayed sources.
- Some scanned PDFs may require OCR quality checks before production use.
- Future improvements could include source freshness monitoring, richer evaluation metrics, access controls for private corpora, and a production-grade observability layer.

## Version Francaise

**SolvaIIRAG est un assistant d'intelligence reglementaire Solvabilite II pret a etre deploye, avec reponses citees, documents sources integres et interface publique Streamlit.**

SolvaIIRAG est une application de Retrieval-Augmented Generation (RAG) construite pour la reglementation assurance. Elle permet de poser des questions pratiques sur Solvabilite II et de consulter les passages sources qui justifient chaque reponse. L'objectif n'est pas de masquer la complexite reglementaire derriere un chatbot, mais de rendre une documentation dense plus searchable, explicable et auditable.

Le projet est pense comme une application portfolio publique : un visiteur peut ouvrir l'application, choisir une question suggeree et obtenir des resultats sources sans configurer de dossier local, importer de fichiers ou fournir une cle API.

### Resume executif

La documentation Solvabilite II est dispersee entre directives, reglements delegues, publications EIOPA, notices des superviseurs locaux et pages de questions-reponses. Pour les equipes risques, actuariat, conformite et conseil, le probleme est rarement l'absence d'information. Le vrai enjeu consiste a trouver rapidement le paragraphe pertinent, comprendre son contexte et conserver une trace fiable vers la source.

SolvaIIRAG repond a ce besoin avec :

- **Un corpus reglementaire integre** dans le repository, couvrant des sources UE, EIOPA, ACPR et BNB/NBB.
- **Un chargement automatique de l'index** au demarrage, pour que la demo publique fonctionne immediatement.
- **Des reponses retrieval-first** qui affichent les passages sources au lieu de s'appuyer uniquement sur une generation non verifiee.
- **Une synthese LLM optionnelle** via `GROQ_API_KEY`, lorsque la cle est disponible.
- **Une interface Streamlit publique** pensee pour montrer rapidement la valeur du projet a un recruteur, un Chief Risk Officer ou un reviewer technique.

### Ce que le projet demontre

| Competence | Illustration dans le projet |
| --- | --- |
| Compréhension du domaine assurance | Les notions SCR, Best Estimate, Risk Margin, ORSA, gouvernance, SFCR/RSR et orientations EIOPA structurent le parcours de demonstration. |
| Conception d'un systeme RAG | Les documents sont charges, decoupes, indexes, recherches, classes et restitues avec citations. |
| Sens produit | L'utilisateur public n'a plus besoin d'indiquer un chemin local ; le corpus et l'index sont integres. |
| Traçabilite | Les reponses restent reliees aux noms de documents, pages, sections et extraits retrouves. |
| Deploiement | L'application est compatible Streamlit Community Cloud et fonctionne sans cle API privee. |

### Fonctionnalites principales

- **Corpus Solvabilite II embarque** dans `Directive/`.
- **Demo publique zero configuration** : pas de chemin local, pas d'upload, pas de cle API obligatoire.
- **Recherche BM25** adaptee aux termes juridiques, numeros d'articles et definitions reglementaires.
- **Parcours hybride** avec Chroma et embeddings multilingues lorsque l'index vectoriel est disponible.
- **Reranking optionnel** avec cross-encoder pour ameliorer l'ordre des sources.
- **Synthese LLM optionnelle** via Groq ; sans cle, l'application reste utilisable en mode recherche citee.
- **Experience centree sur les citations** avec documents, pages, extraits et historique exportable.

### Questions exemples

- Que dit l'article 101 sur le SCR ?
- Comment la Risk Margin est-elle calculee ?
- Quelles sont les exigences de gouvernance sous Solvabilite II ?
- Que dit l'article 45 sur l'ORSA ?
- Comment le Best Estimate est-il defini ?

### Choix techniques

- **BM25 d'abord** car les utilisateurs reglementaires recherchent souvent des formulations exactes, des numeros d'articles et des termes definis.
- **Embeddings multilingues** car le corpus contient des sources en francais et en anglais.
- **Runtime Python statique** pour eviter l'execution dynamique d'un notebook en production.
- **Corpus embarque** pour que l'application soit utilisable par des reviewers externes.
- **Fallback sans LLM** afin que le produit reste utile comme assistant de recherche citee meme sans API payante.

### Evaluation du retrieval

Le composant de recherche a ete evalue sur **127 questions Solvabilite II** couvrant les piliers 1, 2 et 3, la supervision de groupe, les investissements, la reassurance, les modeles internes, la supervision prudentielle et des sujets transversaux.

Resultats globaux :

| Metrique | Valeur |
|---|---:|
| Questions evaluees | 127 |
| Hit@4 | 0.890 |
| Mean MRR | 0.829 |
| Rang moyen du premier hit | 1.19 |
| Couverture moyenne des termes | 0.682 |
| Questions echouees | 14 |

Le Hit@4 de **89,0 %** signifie que le contenu attendu apparait dans les quatre premiers passages pour la plupart des questions. Le rang moyen du premier hit, **1,19**, montre que les resultats pertinents sont generalement classes tres haut.

### Lancer localement

```bash
pip install -r requirements.txt
streamlit run app_solvency_rag_llm.py
```

### Deploiement Streamlit Community Cloud

1. Pousser ce repository sur GitHub.
2. Creer une nouvelle application sur Streamlit Community Cloud.
3. Selectionner la branche `main` et l'entrypoint `app_solvency_rag_llm.py`.
4. Optionnel : ajouter `GROQ_API_KEY` dans les secrets Streamlit pour activer la synthese LLM.

L'application reste utilisable sans cle Groq en mode recherche avec citations.

### Limites et prochaines etapes

- L'application est un assistant de recherche reglementaire, pas un avis juridique.
- La qualite des reponses depend de la completude et de la fraicheur du corpus embarque.
- Les syntheses LLM doivent etre relues a la lumiere des sources affichees.
- Certains PDF scannes peuvent necessiter des controles OCR.
- Les ameliorations futures pourraient inclure un suivi de fraicheur des sources, des metriques d'evaluation plus riches, des controles d'acces pour corpus prives et une observabilite de production.

## Security and Secrets

Do not commit `.streamlit/secrets.toml`, `.env`, API keys, tokens, or private credentials.
