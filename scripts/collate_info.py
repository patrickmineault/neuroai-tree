import pickle

import numpy as np
import pandas as pd


def authorships_to_string(authorships):
    names = [a["author"].get("display_name", "") for a in authorships]
    if len(names) > 5:
        return ", ".join(names[:5]) + ", et al."
    return ", ".join(names)


def main():
    df = pd.read_json("data/processed/works.jsonl", lines=True)

    with open("data/processed/features.pkl", "rb") as f:
        features = pickle.load(f)

    df["neuro_related"] = np.where(features.sum(axis=1) >= 1, 1, 0)

    with open("data/processed/categories.pkl", "rb") as f:
        outputs = pickle.load(f)

    category = np.zeros(df.shape[0], dtype=object)
    category[df["neuro_related"].values == 1] = outputs

    df["category"] = category

    df_ss = pd.read_json("data/processed/semantic_scholar.jsonl", lines=True)
    df_ss["ss_cited_by_count"] = df_ss["result"].map(
        lambda x: x["citationCount"]
    )
    df_ss = df_ss[["id", "ss_cited_by_count"]]

    # Do a left join on the paper ID
    df = df.merge(df_ss, left_on="id", right_on="id", how="left")
    df["author_list"] = df.authorships.map(authorships_to_string)
    df["journal"] = df.primary_location.map(
        lambda x: x["source"]["display_name"]
    )
    df["link"] = df["primary_location"].map(lambda x: x["landing_page_url"])
    df = df[df["neuro_related"] == 1]
    df = df[
        [
            "id",
            "title",
            "publication_year",
            "journal",
            "link",
            "author_list",
            "cited_by_count",
            "category",
            "abstract",
            "ss_cited_by_count",
            "neuro_related",
        ]
    ]

    # Save the final dataframe
    df.to_csv("data/processed/neuroai-works.csv", index=False)


if __name__ == "__main__":
    main()
    main()