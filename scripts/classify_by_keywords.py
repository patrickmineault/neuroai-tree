import pickle

import numpy as np
import pandas as pd

patterns = [
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
    "neurally plausible",
    "neural plausibility",
    "mental patholog-",
    "psychiatr-",
]


def featurize(df):
    F = []
    for feature in patterns:
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
    df = pd.DataFrame(np.array(F).T)
    df.columns = [str(f) for f in patterns]
    return df


def main():
    df = pd.read_json("data/processed/works.jsonl", lines=True)
    features = featurize(df)

    with open("data/processed/features.pkl", "wb") as f:
        pickle.dump(features, f)


if __name__ == "__main__":
    main()
