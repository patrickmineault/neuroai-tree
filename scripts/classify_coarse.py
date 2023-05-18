import json

import numpy as np
import pandas as pd

known_journals = {
    "PLOS Computational Biology",
    "PLoS Comput. Biol.",
    "Trends in Neurosciences",
    "Vision Research",
    "Nature Neuroscience",
    "Journal of Neurophysiology",
    "Neuron",
    "Psychological Science",
    "Psychological Review",
    "Current Opinion in Neurobiology",
    "The Journal of Neuroscience",
    "Cognition",
    "Trends in Cognitive Sciences",
    "Journal of experimental psychology",
    "Cerebral Cortex",
    "The Journal of Physiology",
    "Journal of Physiology",
    "Cognitive Psychology",
    "PLOS Biology",
    "Behavioral and Brain Sciences",
    "Journal of Vision",
    "Nature Reviews Neuroscience",
    "Annual Review of Neuroscience",
    "Experimental Brain Research",
    "American Journal of Psychology",
    "Neurobiology of Learning and Memory",
    "European Journal of Neuroscience",
    "Attention Perception & Psychophysics",
    "Current Biology",
    "Electroencephalography and Clinical Neurophysiology",
    "Progress in Brain Research",
    "Developmental Psychology",
    "NeuroImage",
    "Memory & Cognition",
    "Visual Neuroscience",
    "Journal of Computational Neuroscience",
    "Clinical Neurophysiology",
    "Proceedings of the Annual Meeting of the Cognitive Science Society",
    "Neuropsychologia",
    "Quarterly Journal of Experimental Psychology",
    "Psychonomic Bulletin & Review",
    "Journal of Cognitive Neuroscience",
    "Psychology Press eBooks",
    "Journal of Personality and Social Psychology",
    "Current Directions in Psychological Science",
    "Biological Psychology",
    "Brain and Cognition",
    "Neurobiology of Learning and Memory",
    "Spatial Vision",
    "Human Brain Mapping",
    "Journal of Experimental Psychology: General",
    "Journal of Neural Engineering",
    "Journal of Neuroscience Methods",
    "Journal of Educational Psychology",
    "Brain Research",
    "Journal of Experimental Psychology: Human Perception and Performance",
    "Journal of Experimental Psychology: Learning, Memory and Cognition",
    "Journal of Memory and Language",
    "Cognitive Science",
    "Child Development",
    "Journal of comparative neurology",
    "Psychological Bulletin",
    "Psychophysiology",
    "Brain",
}


keywords = [
    "STDP",
    "NMDA",
    "dendrit-",
    "ion channel",
    "glutamate",
    "neural spik-",
    "neuronal spik-" "spiking neur-",
    "neuromorphic",
    "real neur-",
    "cortex",
    "cortical",
    "neuroscien-",
    "retina-",
    "cochlea-",
    "V1",
    "V2",
    "V4",
    "V5",
    "middle temporal",  # No MT: could also be machine translation
    "inferotemporal",  # No IT: could also be information technology
    "MST",
    "ventral",
    "dorsal",
    "striatum",
    "cerebell-",
    "hippocamp-",
    "amygdala",
    "orbitofrontal",
    "calcium imaging",
    "electrode",
    "electrophysiol-",
    "neurophysiol-",
    "EEG",
    "MEG",
    "fMRI",
    "neuroimaging",
    "functional connectivity",
    "spike sorting",
    "electron microscopy",
    "2AFC",
    "NAFC",
    ["human", "percept-"],
    "psychophysic-",
    "physiolog-",
    "brain",
    "primate",
    "macaque",
    "monkey-",
    "BCI",
    "brain-computer interface",
    "biologically inspired",
    "biologically motivated",
    "biologically plausible",
    "biological plausibility",
    "biological neural",
    "neurally plausible",
    "neural plausibility",
    "mental patholog-",
    "psychiatr-",
]


def get_journal(ref):
    if (
        "primary_location" in ref
        and "source" in ref["primary_location"]
        and ref["primary_location"]["source"] is not None
        and "display_name" in ref["primary_location"]["source"]
    ):
        return ref["primary_location"]["source"]["display_name"]
    elif "venue" in ref:
        return ref["venue"]
    elif "publicationVenue" in ref:
        return ref["publicationVenue"]
    elif "journal" in ref:
        return ref["journal"]
    else:
        return ""


def get_neuro_influenced_citations(df):
    """
    There are two sources of references: OpenAlex and Semantic Scholar.
    OpenAlex is cleaner, but has fewer references. Semantic Scholar has higher coverage
    but is messier. We combine the two sources in a best effort.
    """
    id_to_ss_refs = {}
    with open("data/processed/semantic_scholar.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            id_to_ss_refs[data["id"]] = data["result"]["references"]

    id_to_oa_refs = {}
    with open("data/processed/references.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            id_to_oa_refs[data["id"]] = data["references"]

    for source, id_to_refs in [
        ("ss", id_to_ss_refs),
        ("oa", id_to_oa_refs),
    ]:
        neuro_citations = []
        neuro_journalss = []
        for _, row in df.iterrows():
            if row["id"] not in id_to_refs:
                neuro_citations.append(0)
                neuro_journalss.append("")
                continue

            refs = id_to_refs[row["id"]]
            journals = [get_journal(ref) for ref in refs]

            neuro_journals = []
            for journal in journals:
                if journal in known_journals:
                    neuro_journals.append(journal)

            # Was this paper influenced by multiple neuroscience papers?
            neuro_citations.append(len(neuro_journals))
            neuro_journalss.append(", ".join(neuro_journals))

        df[f"{source}_neuro_citations"] = neuro_citations
        df[f"{source}_cited_journals"] = neuro_journalss

    return df


def keyword_search(df):
    F = []
    for feature in keywords:
        if isinstance(feature, str):
            feature = [feature]
        mask = np.ones(df.shape[0], dtype=bool)
        for f in feature:
            if f.endswith("-"):
                mask = mask & df.abstract.str.contains(" " + f[:-1])
            else:
                mask = mask & (
                    df.abstract.str.contains(" " + f + ".")
                    | df.abstract.str.contains(" " + f + " ")
                    | df.abstract.str.contains(" " + f + "-")
                    | df.abstract.str.contains(" " + f + ":")
                )
        F.append(mask)
    df_ = pd.DataFrame(np.array(F).T)
    df_.columns = [str(f) for f in keywords]
    df["keywords_found"] = (df_.sum(axis=1)).values
    return df


def main():
    df = pd.read_json("data/processed/works.jsonl", lines=True)

    # Find papers with the right kinds of keywords in the abstract.
    df = keyword_search(df)

    # Find papers that cite at least two neuroscience papers.
    df = get_neuro_influenced_citations(df)

    qualified = (
        (df["oa_neuro_citations"] >= 2)
        | (df["ss_neuro_citations"] >= 2)
        | (df["keywords_found"] >= 1)
        | (df["source"] == "manual")
    )

    df["qualified"] = qualified

    df[
        [
            "id",
            "source",
            "qualified",
            "keywords_found",
            "oa_neuro_citations",
            "oa_cited_journals",
            "ss_neuro_citations",
            "ss_cited_journals",
        ]
    ].to_json(
        "data/processed/coarse_classification.jsonl",
        lines=True,
        orient="records",
    )


if __name__ == "__main__":
    main()
