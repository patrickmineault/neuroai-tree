import asyncio
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd

from src.api_request_parallel_processor import process_api_requests_from_file

organization_id = "org-MmmjR4sAdg5Macf3lV3qdGsn"

prompt = """Categorize the paper into one of the following categories. Use a one-letter answer.

A. ML inspired or motivated by brains: solving a general problem in machine learning by taking inspiration from the brain or motivated human and animal intelligence, psychology, or psychophysical findings. Reinforcement learning, artificial neural networks or convolutional neural networks do not qualify unless they are explicitly motivated in the abstract by neuroscience findings.
B. ML strictly in the service of brain research: solving a problem specifically related to the analysis of brain data (e.g. EEG, MEG, fMRI, electrophysiology, calcium imaging) using machine learning techniques
C. ML as models of the brain: comparing the capabilities of machine learning algorithms and humans, ML models as models of the brain
D. General ML with some applicability to brains: solving a general problem in machine learning that could have application to the analysis of brain data, but also in other domains
E. Solving general ML problems which have very little to do with brains
---
{abstract}
"""


def run_requests_parallel(
    requests,
    input_file,
    model="gpt-3.5-turbo",
    model_type="chat",
    word_lists=None,
):
    # Check for previous outputs.
    output_file = "data/interim/prompts-outputs.jsonl"
    with open(output_file, "r") as f:
        outputs = [json.loads(x) for x in f.read().splitlines()]

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

    if model_type == "chat":
        endpoint = "https://api.openai.com/v1/chat/completions"
    else:
        endpoint = "https://api.openai.com/v1/completions"

    if model == "gpt-4":
        max_requests_per_minute = 50
        max_tokens_per_minute = 40000
    else:
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
        if not isinstance(o[1], list):
            clean_outputs.append(o)

    outputs = clean_outputs

    # Reorder outputs.
    output_map = {k: i for i, k in enumerate(keys_all)}
    for i, o in enumerate(outputs):
        if model_type == "chat":
            output_map[json.dumps(o[0])] = o[1]["choices"][0]["message"][
                "content"
            ]
        else:
            output_map[json.dumps(o[0])] = o[1]["choices"][0]["text"]

    output_list = [output_map[x] for x in keys_all]

    outputs = []
    for content in output_list:
        for j in "ABCDEF":
            if content.startswith(j):
                outputs.append(j)
                break
            if j == "F":
                outputs.append("F")
    return outputs


def gather_requests(df, model, model_type):
    requests = []
    df.abstract.fillna("", inplace=True)
    for i, row in df.iterrows():
        abstract = row.fillna("")["abstract"]
        prompt_text = prompt.format(abstract=abstract)

        if model_type == "chat":
            requests.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. You always follow instructions.",
                        },
                        {"role": "user", "content": prompt_text},
                    ],
                    "model": model,
                }
            )
        else:
            requests.append(
                {"prompt": prompt_text, "model": model, "max_tokens": 512}
            )
    return requests


def main():
    model = "gpt-4"  # "gpt-3.5-turbo"
    model_type = "chat"

    df = pd.read_json("data/processed/works.jsonl", lines=True)

    with open("data/processed/features.pkl", "rb") as f:
        features = pickle.load(f)

    df["neuro_related"] = np.where(features.sum(axis=1) >= 1, 1, 0)

    print(df[df["neuro_related"] == 1].shape)

    requests = gather_requests(df[df["neuro_related"] == 1], model, model_type)
    input_file = "data/interim/prompts-input.jsonl"

    outputs = run_requests_parallel(
        requests,
        input_file,
        model,
        model_type,
    )

    with open("data/processed/categories.pkl", "wb") as f:
        pickle.dump(outputs, f)


if __name__ == "__main__":
    main()
