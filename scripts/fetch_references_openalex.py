import json

import pandas as pd
import tqdm
from pyalex import Works


def main():
    columns = [
        "id",
        "ids",
        "title",
        "publication_year",
        "primary_location",
        "authorships",
        "concepts",
        "abstract_inverted_index",
    ]

    with open("data/processed/references.jsonl", "r") as f:
        loaded = {json.loads(x)["id"] for x in f.readlines()}

    df = pd.read_json("data/processed/works.jsonl", lines=True)
    for _, row in tqdm.tqdm(df.iterrows()):
        if row["id"] in loaded:
            continue
        works = (
            Works()
            .select(columns)
            .filter(cited_by=row["id"])
            .paginate(100, n_max=50_000)
        )
        works_processed = []
        for page in works:
            for work in page:
                if work.get("abstract_inverted_index"):
                    abstract = work["abstract"]
                else:
                    abstract = None

                work_processed = dict(work)
                del work_processed["abstract_inverted_index"]
                work_processed["abstract"] = abstract
                works_processed.append(work_processed)

        data = {"id": row["id"], "references": works_processed}
        with open("data/processed/references.jsonl", "a") as f:
            json.dump(data, f)
            f.write("\n")


if __name__ == "__main__":
    main()
