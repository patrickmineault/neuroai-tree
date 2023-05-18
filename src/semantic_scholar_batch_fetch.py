import json

import requests
import semanticscholar
import tqdm
from semanticscholar import SemanticScholar

batch_size_forward = 5
batch_size_backward = 25
batch_size_bi = 10

base_fields = [
    "authors",
    "citationCount",
    "externalIds",
    "fieldsOfStudy",
    "journal",
    "publicationDate",
    "publicationVenue",
    "s2FieldsOfStudy",
    "title",
    "venue",
    "year",
]

citation_fields = ["citations"] + ["citations." + x for x in base_fields]

reference_fields = ["references"] + ["references." + x for x in base_fields]

fields_backward = base_fields + reference_fields

fields_forward = base_fields + citation_fields

fields_bi = base_fields + reference_fields + citation_fields


def chunk_iterator(lst, chunk_size=100):
    """Generator function to iterate through a list in chunks."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def fetch_semantic_scholar_by_ids(ids, fields_list="backward", f=None):
    if fields_list == "backward":
        fields = fields_backward
        batch_size = batch_size_backward
    elif fields_list == "bi":
        fields = fields_bi
        batch_size = batch_size_bi
    elif fields_list == "forward":
        fields = fields_forward
        batch_size = batch_size_forward
    else:
        raise ValueError(f"Unknown fields list {fields_list}")

    sch = SemanticScholar()
    ids = {k: v for k, v in ids.items() if v is not None}
    all_results = []
    for el in tqdm.tqdm(
        chunk_iterator(list(ids.items()), batch_size),
        total=int(len(ids) / batch_size),
    ):
        sub_keys, sub_list = zip(*el)
        try:
            batch_results = sch.get_papers(sub_list, fields=fields)
        except (
            Exception,  # Internal server error is a simple exception.
            TypeError,
            requests.exceptions.ReadTimeout,
            semanticscholar.SemanticScholarException.BadQueryParametersException,
        ):
            batch_results = []
            n_success = 0
            for p in sub_list:
                # Sometimes the thing stalls, in which case we can do it one
                # paper at a time.
                try:
                    all_results += sch.get_papers([p], fields=fields)
                    n_success += 1
                except (
                    Exception,  # Internal server error is a simple exception.
                    TypeError,
                    semanticscholar.SemanticScholarException.BadQueryParametersException,
                    requests.exceptions.ReadTimeout,
                ):
                    batch_results.append(None)
            print(f"Fetch success rate: {n_success / len(sub_list)}")

        batch_results = [
            {"id": k, "result": dict(r)}
            for k, r in zip(sub_keys, batch_results)
            if r is not None
        ]
        if f:
            for r in batch_results:
                f.write(json.dumps(r) + "\n")
        else:
            all_results += batch_results

    return all_results
