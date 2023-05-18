import datetime
import json
import pickle

import cysimdjson
import pandas as pd
import tqdm

from src.cd_index import cd_index


def main():
    # Load up the focal, successors and ancestors for each paper.
    print("Loading data...")
    with open("data/processed/citation-graph.pkl", "rb") as f:
        data = pickle.load(f)
    focii = list(set(data["focii"]))

    print("Loading complete, calculating CD index")
    rows = cd_index(
        focii,
        data["children"],
        data["ancestors"],
        dates=data["dates"],
        time_window=datetime.timedelta(days=365 * 5 + 10),
    )

    df = pd.DataFrame(rows).set_index("focus")

    parser = cysimdjson.JSONParser()
    # Now grab the metadata for each paper.
    rows = []
    seen = set()
    with open("data/processed/semantic_scholar.jsonl") as f:
        for line in tqdm.tqdm(f):
            data = parser.parse_string(line)["result"]
            focal_id = data["paperId"]

            if focal_id not in focii or focal_id in seen:
                continue

            seen.add(focal_id)

            data = json.loads(line)["result"]
            data = {**data, **df.loc[focal_id].to_dict()}
            rows.append(data)

    df_all = pd.DataFrame(rows)
    df_all.to_parquet("data/processed/cd-index.parquet", index=False)


if __name__ == "__main__":
    main()
