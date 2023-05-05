import pandas as pd
import streamlit as st
from transformers import pipeline


# Load your pandas file with the paper information
def load_data():
    df = pd.read_csv("data/processed/neuroai-works.csv")
    df.sort_values("ss_cited_by_count", ascending=False, inplace=True)
    return df


@st.cache_data()
def get_data():
    return load_data()


categories = {
    "A": "Neuro-inspired AI",
    "B": "AI for neuro",
    "C": "AI compared to neuro",
    "D": "General AI, some neuro",
    "E": "Not neuro related",
}


# Main app
def main():
    df = get_data()

    # Left sidebar
    with st.sidebar:
        st.header("NeuroAI paper browser")
        category = st.selectbox(
            "Category",
            ["A", "C", "B", "D", "E"],
            format_func=lambda x: categories[x],
        )
        paper_titles = df.loc[df.category == category, "title"].tolist()
        selected_paper_index = st.number_input(
            f"Choose a paper (0-{len(paper_titles)-1}):",
            min_value=0,
            max_value=len(paper_titles) - 1,
        )

    # Right side: Display paper information
    selected_paper = df[df.category == category].iloc[selected_paper_index]

    st.markdown(f"# {selected_paper['title']}")  # Paper title as an h1
    st.write(
        f"Number of Citations: {int(selected_paper['ss_cited_by_count'])}"
    )
    st.write(f"Publication Year: {selected_paper['publication_year']}")
    st.write(f"Author List: {selected_paper['author_list']}")
    st.write(f"Journal: {selected_paper['journal']}")
    st.write(f"Link: {selected_paper['link']}")
    abstract_highlighted = selected_paper["abstract_highlighted"]
    st.markdown(f"Abstract: {abstract_highlighted}")


if __name__ == "__main__":
    main()
