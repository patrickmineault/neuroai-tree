import pandas as pd


def authorships_to_string(authorships):
    names = [a["author"].get("display_name", "") for a in authorships]
    if len(names) > 5:
        return ", ".join(names[:5]) + ", et al."
    return ", ".join(names)


def protect(x):
    if x is None:
        return []
    return x


def main():
    df = pd.read_json("data/processed/works.jsonl", lines=True)
    df = df.rename(columns={"source": "origin"})

    df_ss = pd.read_json("data/processed/semantic_scholar.jsonl", lines=True)
    df_ss["references"] = df_ss["result"].map(lambda x: x["references"])
    df_ss["ssid"] = df_ss["result"].map(lambda x: x["paperId"])
    df_ss = df_ss[["id", "references", "ssid"]]

    # Do a left join on the paper ID
    df = df.merge(df_ss, left_on="id", right_on="id", how="left")

    # Get the most cited paper in order.
    references = dict()
    for _, row in df.iterrows():
        if not isinstance(row.references, list):
            continue

        for ref in row.references:
            if ref["paperId"] in references:
                references[ref["paperId"]]["citations"] += 1
            else:
                references[ref["paperId"]] = {
                    "id": ref["paperId"],
                    "author_list": authorships_to_string(ref["authors"]),
                    "title": ref["title"],
                    "journal": ref["journal"],
                    "venue": ref["venue"],
                    "s2FieldsOfStudy": {
                        x["category"] for x in protect(ref["s2FieldsOfStudy"])
                    },
                    "fieldsOfStudy": set(protect(ref["fieldsOfStudy"])),
                    "citations": 1,
                }

    df_refs = pd.DataFrame(references.values())
    df_refs.sort_values(by="citations", ascending=False, inplace=True)
    df_refs.to_csv("data/processed/citation-references.csv", index=False)


if __name__ == "__main__":
    main()
    main()
