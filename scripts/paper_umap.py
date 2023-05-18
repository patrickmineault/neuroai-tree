import pickle
import textwrap

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from paper_browser import write_details
from streamlit_plotly_events import plotly_events

categories = {
    "A": "Neuro-inspired AI",
    "B": "AI for neuro",
    "C": "AI compared to neuro",
    "D": "General AI, some neuro",
    "E": "Not neuro related",
    "F": "Not classified",
    "*": "All",
}


# Load your pandas file with the paper information
def load_data():
    df = pd.read_csv("data/processed/neuroai-works.csv")
    df.sort_values("ss_cited_by_count", ascending=False, inplace=True)
    df = df[df["openai_category"].isin(["A", "B", "C"])]

    with open("data/processed/umap_labels.pickle", "rb") as f:
        labels = pickle.load(f)

    return df, labels


@st.cache_resource()
def make_plot(df_, labels):
    fig = px.line(x=[1], y=[1])
    fig = px.scatter(
        data_frame=df_,
        x="umap_x",
        y="umap_y",
        color="label",
        size="log_ss_cited_by_count",
        hover_name="title_wrapped",
        hover_data={
            "title": False,
            "title_wrapped": False,
            "umap_x": False,
            "umap_y": False,
            "log_ss_cited_by_count": False,
            "authors": True,
        },
        color_discrete_sequence=px.colors.qualitative.G10,
        height=600,
        width=np.inf,
    )

    for i, label in enumerate(labels["labels"]):
        # Add the annotation to the plot
        fig.add_annotation(
            x=labels["label_centers"][i][0],
            y=labels["label_centers"][i][1],
            text="<br>".join(textwrap.wrap(label, 30)),
            showarrow=False,
            font=dict(color="white", size=12),  # Set the font color and size
            align="center",  # Horizontal alignment ('left', 'center', or 'right')
            valign="middle",  # Vertical alignment ('top', 'middle', or 'bottom')
            opacity=0.5,  # Adjust the opacity of the background text
            bgcolor="#000000",  # Set the background color of the text
            borderpad=3,
        )

    fig.update_layout(hovermode="closest")
    fig.update(layout_showlegend=False)
    return fig


@st.cache_data()
def get_data():
    return load_data()


def main():
    st.set_page_config(layout="wide")
    st.title("NeuroAI paper browser")
    st.write("Click on a paper to see details at the bottom.")

    df, labels = get_data()
    df_ = df.copy()
    df_["log_ss_cited_by_count"] = np.log(
        df_["ss_cited_by_count"].fillna(0) + 1
    )
    df_["title_wrapped"] = df_["title"].str.wrap(50).str.replace("\n", "<br>")
    df_["authors"] = df_["author_list"].str.wrap(50).str.replace("\n", "<br>")
    df_["label"] = df_["openai_category"].map(lambda x: categories[x])
    df_ = df_.sort_values("label").reset_index(drop=True)
    selected_paper_index = 0

    fig = make_plot(df_, labels)

    selected_points = plotly_events(fig, click_event=True)
    label_list = sorted(df_.label.unique().tolist())
    if len(selected_points) == 1:
        selected_curve = selected_points[0]["curveNumber"]
        selected_paper_index = selected_points[0]["pointIndex"]
        df__ = df_[df_.label == label_list[selected_curve]]
        selected_paper = df__.iloc[selected_paper_index]

        write_details(selected_paper, categories)


if __name__ == "__main__":
    main()
