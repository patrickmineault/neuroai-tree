import json
import pickle

import Levenshtein
import pandas as pd
import semanticscholar
import tqdm
from collections_extended import bag
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
                    all_results.append(None)
            print(n_success)

    assert len(all_results) == len(
        ids
    ), f"Length mismatch, {len(all_results)} != {len(ids)}"
    return [
        {"id": k, "result": dict(r)}
        for k, r in zip(list(ids.keys()), all_results)
    ]


def fetch_semantic_scholar_by_title(title):
    sch = SemanticScholar()
    title = title.replace("-", " ").replace("\n", " ")
    if ":" in title:
        title_l, title_r = title.split(":")
        if len(title_l) > len(title_r) // 2:
            title = title_l
        elif len(title_r) > len(title_l) // 2:
            title = title_r
    try:
        results = sch.search_paper(
            title,
            fields=["title", "venue"],
            limit=5,
        )
    except KeyError:
        return None

    n = 0
    for result in results:
        rtitle = result.title
        distance = Levenshtein.distance(rtitle.lower(), title.lower())
        if result.venue is not None and distance < 5:
            return result["paperId"]
        else:
            # Use a bag instead.
            bag_query = bag(title.lower())
            bag_result = bag(rtitle.lower())

            if len(bag_query & bag_result) > 0.95 * min(
                len(bag_query), len(bag_result)
            ):
                return result["paperId"]

        n += 1
        if n > 5:
            break

    print("No match found for", title)
    return None


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
        if row["id"] in paper_ids and row["id"] is not None:
            continue

        # Resolve Semantic Scholar ID by title
        try:
            paper_id = fetch_semantic_scholar_by_title(
                row["title"].replace("\n", " ")
            )
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
