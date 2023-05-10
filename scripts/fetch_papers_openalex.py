import json
import shutil

import Levenshtein
import pandas as pd
import pyalex
import tqdm
from collections_extended import bag
from pyalex import Sources, Works

# Polite pool.
pyalex.config.email = "patrick.mineault@gmail.com"

# Replace this list with the ORCID IDs of the authors you want to fetch data for
openalex_ids = [
    "S4306420609",  # Neural Information Processing Systems
    "S4306419644",  # ICML
    "S4306419637",  # ICLR
    "S4210191458",  # AAAI
    "S4306417987",  # CVPR (except 2022)
    "S4363607795",  # CVPR 2009
    "S4306417987",  # Late 2010's CVPR
    "S4363607701",  # CVPR 2022
    "S4306419272",  # ICCV (except 2021)
    "S4363607764",  # ICCV 2021
]


def protect(name):
    if name:
        return name
    return ""


# Function to fetch abstracts for authors with given ORCID IDs
def fetch_abstracts_by_title(titles):
    encountered = set()

    for title in tqdm.tqdm(titles):
        works = (
            Works()
            .filter(
                title={"search": title.replace(",", " ").replace("&", "and")}
            )
            .get()
        )

        if not works:
            continue

        work = works[0]

        distance = Levenshtein.distance(work["title"].lower(), title.lower())
        if distance >= 5:
            # Try as a bag
            bag_query = bag(title.lower())
            bag_result = bag(work["title"].lower())

            if not (
                len(bag_query & bag_result)
                > 0.95 * min(len(bag_query), len(bag_result))
            ):
                continue

        if work["id"] in encountered:
            continue
        if work.get("abstract_inverted_index"):
            abstract = work["abstract"]
        else:
            abstract = None

        encountered.add(work["id"])

        with open("data/interim/works.jsonl", "a") as f:
            row_processed = dict(work)
            del row_processed["abstract_inverted_index"]
            row_processed["abstract"] = abstract
            row_processed["source"] = "manual"
            json.dump(row_processed, f)
            f.write("\n")
    return encountered


def fetch_abstracts_by_conference(openalex_ids, encountered):
    for openalex_id in tqdm.tqdm(openalex_ids):
        source = Sources()[openalex_id]
        pages = (
            Works()
            .filter(primary_location={"source": {"id": source["id"]}})
            .paginate(100, n_max=50_000)
        )

        for page in pages:
            for work in page:
                if work["id"] in encountered:
                    continue

                if work.get("abstract_inverted_index"):
                    abstract = work["abstract"]
                else:
                    abstract = None

                encountered.add(work["id"])

                with open("data/interim/works.jsonl", "a") as f:
                    row_processed = dict(work)
                    del row_processed["abstract_inverted_index"]
                    row_processed["abstract"] = abstract
                    row_processed["source"] = "conference-list"
                    json.dump(row_processed, f)
                    f.write("\n")
    return encountered


if __name__ == "__main__":
    # Fetch the abstracts
    # Clear the file
    with open("data/interim/works.jsonl", "w") as f:
        f.write("")

    df_manual = pd.read_csv("data/raw/manual-references.csv")

    encountered = fetch_abstracts_by_title(df_manual.Title.values.tolist())
    print(f"Fetched {len(encountered)} abstracts from manual references")
    encountered = fetch_abstracts_by_conference(openalex_ids, encountered)

    print(f"{len(encountered)} abstracts saved to data/processed/works.jsonl")

    # Copy the works.jsonl file to the processed folder atomically.
    shutil.copy("data/interim/works.jsonl", "data/processed/works.jsonl")
