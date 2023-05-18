import json
import pickle

import Levenshtein
import pandas as pd
import tqdm
from collections_extended import bag
from semanticscholar import SemanticScholar

from src.semantic_scholar_batch_fetch import fetch_semantic_scholar_by_ids


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
        except Exception:
            paper_ids[row["id"]] = None

        with open("data/processed/ids.pkl", "wb") as f:
            pickle.dump(paper_ids, f)

    return paper_ids


def main():
    df = pd.read_json("data/processed/works.jsonl", lines=True)
    ids = resolve_semantic_scholar_ids(df)

    f = open("data/processed/semantic_scholar.jsonl", "w")
    fetch_semantic_scholar_by_ids(ids, "bi", f)
    f.close()


if __name__ == "__main__":
    main()
