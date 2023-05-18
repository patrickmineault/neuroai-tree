import json
import pickle
import random

import pandas as pd

from src.semantic_scholar_batch_fetch import fetch_semantic_scholar_by_ids


def get_forward_backward_ids(df):
    try:
        with open("data/processed/focii.pkl", "rb") as f:
            focii = pickle.load(f)
            use_existing = True
    except FileNotFoundError:
        focii = []
        use_existing = False

    children_to_fetch = {}
    ancestors_to_fetch = {}

    with open("data/processed/semantic_scholar.jsonl") as f:
        for line in f:
            data = json.loads(line)
            if (use_existing and data["result"]["paperId"] in focii) or (
                not use_existing
                and (
                    data["id"] in df.id.values.tolist()
                    or (
                        data["result"]["citationCount"] < 2000
                        and random.random() < 0.1
                    )
                )
            ):
                if not use_existing:
                    focii.append(data["result"]["paperId"])
                for ref in data["result"]["citations"]:
                    children_to_fetch[ref["paperId"]] = ref["paperId"]
                for ref in data["result"]["references"]:
                    ancestors_to_fetch[ref["paperId"]] = ref["paperId"]

    if not use_existing:
        with open("data/processed/focii.pkl", "wb") as f:
            pickle.dump(focii, f)

    return children_to_fetch, ancestors_to_fetch


def main():
    # Create a master list of papers to fetch from Semantic Scholar, based on
    # the papers that we want to examine for centrality (CD index).
    df = pd.read_csv("data/processed/neuroai-works.csv")

    children_to_fetch, ancestors_to_fetch = get_forward_backward_ids(df)

    try:
        f = open("data/processed/ancestors.jsonl", "r")
        for line in f:
            data = json.loads(line)
            if data["id"] in ancestors_to_fetch:
                del ancestors_to_fetch[data["id"]]
    except FileNotFoundError:
        pass

    print(f"Parents to fetch: {len(ancestors_to_fetch)}")
    f = open("data/processed/ancestors.jsonl", "a")
    fetch_semantic_scholar_by_ids(ancestors_to_fetch, "forward", f)
    f.close()


if __name__ == "__main__":
    main()
