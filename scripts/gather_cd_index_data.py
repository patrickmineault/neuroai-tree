import collections
import datetime
import json
import pickle

import cysimdjson
import tqdm


def grab_date(x):
    if "publicationDate" in x and x["publicationDate"] is not None:
        return datetime.datetime.strptime(x["publicationDate"], "%Y-%m-%d")
    # Return a mid-year date.
    return datetime.datetime(int(x["year"]), 6, 30)


def main():
    # Load up the focal, successors and ancestors for each paper.
    with open("data/processed/focii.pkl", "rb") as f:
        focii = pickle.load(f)

    children = collections.defaultdict(set)
    ancestors = collections.defaultdict(set)

    parser = cysimdjson.JSONParser()

    # Load up data from the focii, the children and the ancestors.
    dates = {}
    with open("data/processed/semantic_scholar.jsonl") as f:
        for line in tqdm.tqdm(f):
            data = json.loads(line)["result"]
            if (
                data["paperId"] is None
                or "year" not in data
                or data["year"] is None
            ):
                continue
            focal_id = data["paperId"]

            if focal_id not in focii:
                continue

            citations = [
                x
                for x in data["citations"]
                if x["paperId"] is not None
                and "year" in x
                and x["year"] is not None
            ]

            children[focal_id] |= {x["paperId"] for x in citations}
            for x in citations:
                ancestors[x["paperId"]].add(focal_id)
                dates[x["paperId"]] = grab_date(x)

            references = [
                x
                for x in data["references"]
                if x["paperId"] is not None
                and "year" in x
                and x["year"] is not None
            ]
            ancestors[focal_id] |= {x["paperId"] for x in references}
            for x in references:
                children[x["paperId"]].add(focal_id)
                dates[x["paperId"]] = grab_date(x)

            dates[focal_id] = grab_date(data)

    with open("data/processed/ancestors.jsonl") as f:
        for line in tqdm.tqdm(f):
            data = parser.parse_string(line)["result"]
            if (
                data["paperId"] is None
                or "year" not in data
                or data["year"] is None
            ):
                continue
            focal_id = data["paperId"]

            citations = [
                x
                for x in data["citations"]
                if x["paperId"] is not None
                and "year" in x
                and x["year"] is not None
            ]
            children[focal_id] |= {x["paperId"] for x in citations}
            for x in citations:
                ancestors[x["paperId"]].add(focal_id)
                dates[x["paperId"]] = grab_date(x)

            dates[focal_id] = grab_date(data)

    # Save the data to disk.
    with open("data/processed/citation-graph.pkl", "wb") as f:
        pickle.dump(
            {
                "focii": focii,
                "children": children,
                "ancestors": ancestors,
                "dates": dates,
            },
            f,
        )


if __name__ == "__main__":
    main()
