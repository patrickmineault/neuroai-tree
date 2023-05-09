import json
import pickle

import pandas as pd
import semanticscholar
import tqdm
from semanticscholar import SemanticScholar


def chunk_iterator(lst, chunk_size=100):
    """Generator function to iterate through a list in chunks."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


fields = [
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
    "references.publicationVenue",
    "references.journal",
    "references.year",
    "references.publicationDate",
    "references.fieldsOfStudy",
    "references.s2FieldsOfStudy",
    "embedding",
]


def fetch_semantic_scholar_by_ids(ids):
    sch = SemanticScholar()
    ids = {k: v for k, v in ids.items() if v is not None}
    all_results = []
    for i, sub_list in tqdm.tqdm(
        enumerate(chunk_iterator(list(ids.values()))),
        total=int(len(ids) / 100),
    ):
        try:
            all_results += sch.get_papers(sub_list, fields=fields)
        except TypeError:
            n_success = 0
            for p in sub_list:
                # Sometimes the thing stalls, in which case we can do it one
                # paper at a time.
                try:
                    all_results += sch.get_papers([p], fields=fields)
                    n_success += 1
                except (
                    semanticscholar.SemanticScholarException.BadQueryParametersException
                ):
                    continue
            print(n_success)

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
        try:
            paper_id = fetch_semantic_scholar_by_title(row["title"])
            paper_ids[row["id"]] = paper_id
        except Exception as e:
            paper_ids[row["id"]] = None

        with open("data/processed/ids.pkl", "wb") as f:
            pickle.dump(paper_ids, f)

    return paper_ids


def main():
    df = pd.read_json("data/processed/works.jsonl", lines=True)
    ids = resolve_semantic_scholar_ids(df)

    results = fetch_semantic_scholar_by_ids(ids)
    with open("data/processed/semantic_scholar.jsonl", "w") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")


if __name__ == "__main__":
    main()
