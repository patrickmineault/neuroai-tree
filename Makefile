.PHONY: run_paper_browser run_umap

all: data/processed/neuroai-works.csv

data/raw/manual-references.csv: scripts/fetch_manual_references.py
	python scripts/fetch_manual_references.py

data/processed/works.jsonl: scripts/fetch_papers_openalex.py data/raw/manual-references.csv
	python scripts/fetch_papers_openalex.py

data/processed/semantic_scholar.jsonl: data/processed/works.jsonl scripts/fetch_papers_semantic_scholar.py
	python scripts/fetch_papers_semantic_scholar.py

data/processed/references.jsonl: data/processed/works.jsonl scripts/fetch_references_openalex.py
	python scripts/fetch_references_openalex.py

data/processed/coarse_classification.jsonl: data/processed/works.jsonl data/processed/semantic_scholar.jsonl data/processed/references.jsonl scripts/classify_coarse.py
	python scripts/classify_coarse.py

data/processed/categories.jsonl: data/processed/works.jsonl data/processed/coarse_classification.jsonl scripts/classify_via_openai.py
	python scripts/classify_via_openai.py

data/processed/neuroai-works.csv: data/processed/works.jsonl data/processed/semantic_scholar.jsonl data/processed/references.jsonl data/processed/coarse_classification.jsonl data/processed/categories.jsonl scripts/collate_info.py
	python scripts/collate_info.py

run_paper_browser: 
	streamlit run scripts/paper_browser.py

run_umap: 
	streamlit run scripts/paper_umap.py