import pickle

import openai
import pandas as pd
import tqdm
from sklearn.cluster import KMeans


def request_label(titles):
    titles = "\n".join(titles)
    prompt = f"Give a 2-5 word label that summarizes the common topic of these abstracts. Avoid vague labels like 'artificial intelligence', 'machine learning', 'neuroscience' and 'deep learning'\n\n {titles}"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion["choices"][0]["message"]["content"].split(".")[0]


def main():
    df = pd.read_csv("data/processed/neuroai-works.csv")
    df.sort_values("ss_cited_by_count", ascending=False, inplace=True)
    df = df[df["openai_category"].isin(["A", "B", "C"])]

    kmeans = KMeans(n_clusters=25, random_state=0, n_init="auto").fit(
        df[["umap_x", "umap_y"]]
    )
    labels = kmeans.fit_predict(df[["umap_x", "umap_y"]].values)

    label_map = []
    for label in tqdm.tqdm(range(kmeans.cluster_centers_.shape[0])):
        titles = df.iloc[labels == label].title.values.tolist()
        label_name = request_label(titles)
        label_map.append(label_name)

    label_info = {
        "label_centers": kmeans.cluster_centers_,
        "labels": label_map,
        "paper_labels": [label_map[x] for x in labels],
    }

    with open("data/processed/umap_labels.pickle", "wb") as f:
        pickle.dump(label_info, f)


if __name__ == "__main__":
    main()
