# neuroai-tree

NeuroAI tree of ideas project. Fetch data from OpenAlex for different conferences and classify it by area of interest to figure out how neuro has influenced AI over time.

Scripts:

* `python scripts/fetch_papers_openalex.py`: fetches papers on OpenAlex
* `python scripts/fetch_papers_semantic_scholar.py`: fetches same papers on SemanticScholar to get full list of references from each paper.
* `python scripts/fetch_references_openalex.py`: fetches references on OpenAlex. Generally has cleaner reference lists.
* `python scripts/classify_by_keywords.py`: do a coarse classification of papers via keywords
* `python scripts/classify_via_openai.py`: do a fine-grained classification of those papers idenfied by keywords via OpenAI
* `python scripts/collate.py`: get info from all these sources and put them into one big dataframe.
* `streamlit run scripts/paper_browser.py`: run an interactive visualization showing papers in every class

This can all be run via `make all`.

Secondary analysis: `scripts/View Influence Over Time.ipynb`
