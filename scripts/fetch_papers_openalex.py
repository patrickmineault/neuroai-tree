import json

import pandas as pd
import pyalex
import tqdm
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
def fetch_abstracts(openalex_ids):
    abstract_data = []

    with open("data/processed/works.jsonl", "w") as f:
        f.write("")

    for openalex_id in tqdm.tqdm(openalex_ids):
        source = Sources()[openalex_id]
        pages = (
            Works()
            .filter(primary_location={"source": {"id": source["id"]}})
            .paginate(100, n_max=50_000)
        )

        for page in pages:
            for work in page:
                if work.get("abstract_inverted_index"):
                    abstract = work["abstract"]
                else:
                    abstract = None

                publication_name = None
                if work["primary_location"]:
                    if work["primary_location"]["source"]:
                        publication_name = work["primary_location"]["source"][
                            "display_name"
                        ]

                row = {
                    "source": source["display_name"],
                    "authors": ", ".join(
                        [
                            protect(a["author"]["display_name"])
                            for a in work["authorships"]
                        ]
                    ),
                    "date": work["publication_date"],
                    "title": work["title"],
                    "publication_name": publication_name,
                    "total_citations": work["cited_by_count"],
                    "abstract": abstract,
                }
                abstract_data.append(row)

                with open("data/processed/works.jsonl", "a") as f:
                    row_processed = dict(work)
                    del row_processed["abstract_inverted_index"]
                    row_processed["abstract"] = abstract
                    json.dump(row_processed, f)
                    f.write("\n")

    return abstract_data


if __name__ == "__main__":
    # Fetch the abstracts
    abstracts = fetch_abstracts(openalex_ids)

    # Create a DataFrame and save it as a CSV file
    df = pd.DataFrame(abstracts)
    df.to_csv("data/raw/abstracts.csv", index=False)

    print("Abstracts saved to abstracts.csv")
