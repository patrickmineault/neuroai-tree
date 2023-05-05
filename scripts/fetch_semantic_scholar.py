import json
import pickle

import pandas as pd
import tqdm
from semanticscholar import SemanticScholar


def chunk_iterator(lst, chunk_size=100):
    """Generator function to iterate through a list in chunks."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def fetch_semantic_scholar_by_ids(ids):
    sch = SemanticScholar()
    ids = {k: v for k, v in ids.items() if v is not None}
    all_results = []
    for sub_list in tqdm.tqdm(chunk_iterator(list(ids.values()))):
        all_results += sch.get_papers(
            sub_list,
            fields=[
                "title",
                "venue",
                "citationCount",
                "fieldsOfStudy",
                "s2FieldsOfStudy",
                "references",
                "references.abstract",
                "references.externalIds",
                "references.title",
                "references.venue",
                "references.year",
                "references.fieldsOfStudy",
                "references.s2FieldsOfStudy",
            ],
        )

    return [
        {"id": k, "result": dict(r)}
        for k, r in zip(list(ids.keys()), all_results)
    ]


def fetch_semantic_scholar_by_title(title):
    sch = SemanticScholar()
    try:
        results = sch.search_paper(
            title.replace("-", " "),
            fields=["title", "venue"],
            limit=5,
        )
    except KeyError:
        return None

    n = 0
    for result in results:
        if result.venue is not None:
            return result["paperId"]
        n += 1
        if n > 5:
            break

    raise Exception(f"Could not find paper with title {title}")


def resolve_semantic_scholar_ids(df):
    paper_ids = {}
    try:
        with open("data/processed/ids.pkl", "rb") as f:
            paper_ids = pickle.load(f)
    except FileNotFoundError:
        paper_ids = {}

    for i, row in tqdm.tqdm(
        df.iterrows(), total=len(df), desc="Resolving Semantic Scholar IDs"
    ):
        if row["id"] in paper_ids:
            continue

        # Resolve Semantic Scholar ID by title
        paper_id = fetch_semantic_scholar_by_title(row["title"])
        paper_ids[row["id"]] = paper_id

        with open("data/processed/ids.pkl", "wb") as f:
            pickle.dump(paper_ids, f)

    return paper_ids


def main():
    df = pd.read_csv("data/processed/neuroai-works.csv")
    ids = resolve_semantic_scholar_ids(df)

    results = fetch_semantic_scholar_by_ids(ids)
    with open("data/processed/semantic_scholar.jsonl", "w") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")


if __name__ == "__main__":
    main()
