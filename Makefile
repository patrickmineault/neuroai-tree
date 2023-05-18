.PHONY: vis_papers vis_umap calc_umap calc_neuroai calc_cdindex

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

data/processed/neuroai-works.csv data/processed/all-works.csv: data/processed/works.jsonl data/processed/semantic_scholar.jsonl data/processed/references.jsonl data/processed/coarse_classification.jsonl data/processed/categories.jsonl scripts/collate_info.py
	python scripts/collate_info.py

data/processed/ancestors.jsonl: data/processed/neuroai-works.csv scripts/fetch_ancestors.py
	python scripts/fetch_ancestors.py

data/processed/citation-graph.pkl: data/processed/focii.pkl data/processed/semantic_scholar.jsonl data/processed/ancestors.jsonl scripts/gather_cd_index_data.py
	python scripts/gather_cd_index_data.py

data/processed/cd-index.parquet: data/processed/citation-graph.pkl data/processed/semantic_scholar.jsonl scripts/calculate_cd_index.py
	python scripts/calculate_cd_index.py

data/processed/umap_labels.pickle data/processed/neuroai-works-umap.csv: data/processed/neuroai-works.csv scripts/compute_umap_labels.py
	python scripts/compute_umap_labels.py

calc_cdindex: data/processed/cd-index.parquet
	@echo "CD index gathered"

calc_neuroai: data/processed/neuroai-works.csv
	@echo "NeuroAI works gathered"

calc_umap: data/processed/umap_labels.pickle
	@echo "UMAP labels computed"

vis_papers: 
	streamlit run scripts/paper_browser.py

vis_umap: 
	streamlit run scripts/paper_umap.py