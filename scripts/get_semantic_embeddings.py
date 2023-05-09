import asyncio
import json
import logging
import os

import pandas as pd

from src.api_request_parallel_processor import process_api_requests_from_file

organization_id = "org-MmmjR4sAdg5Macf3lV3qdGsn"


def run_requests_parallel(requests):
    # Check for previous outputs.
    input_file = "data/interim/embeddings-inputs.jsonl"
    output_file = "data/interim/embeddings-outputs.jsonl"
    try:
        with open(output_file, "r") as f:
            outputs = [json.loads(x) for x in f.read().splitlines()]
    except FileNotFoundError:
        outputs = []

    clean_outputs = []
    for o in outputs:
        if not isinstance(o[1], list):
            clean_outputs.append(o)

    outputs = clean_outputs

    keys_all = [json.dumps(x) for x in requests]
    keys = [json.dumps(x) for x in requests]

    for o in outputs:
        key = json.dumps(o[0])
        if key in keys:
            keys.remove(key)

    print(f"Fetching {len(keys)} keys")

    with open(input_file, "w") as f:
        f.write("\n".join(keys))

    with open(input_file, "r") as f:
        keys = [x.strip() for x in f.readlines()]

    endpoint = "https://api.openai.com/v1/embeddings"

    max_requests_per_minute = 900
    max_tokens_per_minute = 90000

    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=input_file,
            api_key=os.environ.get("OPENAI_API_KEY"),
            organization_id=organization_id,
            save_filepath=output_file,
            request_url=endpoint,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name="cl100k_base",
            max_attempts=5,
            logging_level=logging.INFO,
        )
    )

    with open(output_file, "r") as f:
        outputs = [json.loads(x) for x in f.read().splitlines()]

    clean_outputs = []
    for o in outputs:
        if not isinstance(o[1], list):  # This was an error.
            clean_outputs.append(o)

    outputs = clean_outputs

    # Reorder outputs.
    output_map = {k: i for i, k in enumerate(keys_all)}
    for i, o in enumerate(outputs):
        output_map[json.dumps(o[0])] = o[1]["data"][0]["embedding"]

    outputs = [{"id": x, "embedding": output_map[x]} for x in keys_all]

    return outputs


def gather_requests(df, model, model_type):
    requests = []
    df.abstract.fillna("", inplace=True)
    for i, row in df.iterrows():
        abstract = row.fillna("")["abstract"]
        requests.append({"input": abstract, "model": model})
    return requests


def main():
    model = "text-embedding-ada-002"
    model_type = "embedding"

    df = pd.read_json("data/processed/works.jsonl", lines=True)
    df = df[~df.id.duplicated()]

    # Get the classification from OpenAI
    df_class = pd.read_json("data/processed/categories.jsonl", lines=True)
    df = df.merge(df_class, on="id")
    df = df[~df.openai_category.isna()]

    assert (df.shape[0] < 3000) and (df.shape[0] > 1500)

    requests = gather_requests(df, model, model_type)
    embeddings = run_requests_parallel(requests)

    df_ = pd.DataFrame(embeddings)
    df_["id"] = df["id"].values
    df_.to_json(
        "data/processed/semantic_embeddings.jsonl",
        orient="records",
        lines=True,
    )


if __name__ == "__main__":
    main()
