import numpy as np
import pandas as pd
import streamlit as st


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


def protected_round(x):
    if isinstance(x, float) and not np.isnan(x):
        return round(x)
    return x


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
        text_filter = st.text_input("Text search")

        if text_filter:
            df_ = df[
                df.title.str.contains(text_filter, case=False)
                | df.author_list.str.contains(text_filter, case=False)
                | df.abstract.str.contains(text_filter, case=False)
            ]
        else:
            df_ = df.copy()

        year_range = st.slider(
            "Publication year range", 1980, 2023, (1980, 2023), 1
        )

        df_ = df_[df_.publication_year.between(*year_range)]
        df_ = df_[df_.openai_category == category]

        n_papers = df_.shape[0]
        selected_paper_index = st.number_input(
            f"Choose a paper (0-{n_papers-1}):",
            min_value=0,
            max_value=n_papers - 1,
        )

    # Right side: Display paper information
    selected_paper = df_.iloc[selected_paper_index]
    clean_title = selected_paper["title"].replace("\n", " ")
    st.markdown(f"# {clean_title}")  # Paper title as an h1
    st.write(
        f"Number of Citations: {protected_round(selected_paper['ss_cited_by_count'])}"
    )
    st.write(f"Included because: {selected_paper['reason']}")
    st.write(f"Publication Year: {selected_paper['publication_year']}")
    st.write(f"Author List: {selected_paper['author_list']}")
    st.write(f"Journal: {selected_paper['journal']}")
    st.write(f"Link: {selected_paper['link']}")
    abstract_highlighted = selected_paper["abstract_highlighted"]
    st.markdown(f"Abstract: {abstract_highlighted}")
    st.write(
        f"Neuro journals cited (OpenAlex): {selected_paper['oa_cited_journals']}"
    )
    st.write(
        f"Neuro journals cited (Semantic Scholar): {selected_paper['ss_cited_journals']}"
    )
    oa_url = (
        "https://api.openalex.org/works/" + selected_paper["id"].split("/")[-1]
    )
    st.write(f"OAID: {oa_url}")
    ss_url = "https://www.semanticscholar.org/paper/" + selected_paper["ssid"]
    st.write(f"SSID: {ss_url}")


if __name__ == "__main__":
    main()
