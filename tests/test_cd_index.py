import collections
import datetime

import cdindex
import numpy as np
import numpy.testing as npt
import pandas as pd

from src.cd_index import cd_index


def calculate_ancestors(children):
    ancestors = collections.defaultdict(set)
    for parent, children in children.items():
        for child in children:
            ancestors[child].add(parent)
    return ancestors


def test_cd_index_dated():
    focus = [0]
    # Examples from Figure 1 of Funk & Owen-Smith (2017)
    # http://russellfunk.org/cdindex/static/papers/funk_ms_2017.pdf
    children = {
        0: {1, 2, 3, 4, 5, 6, 7},
        -1: {0},
        -2: {0},
        -3: {0},
        -4: {0, 7},
    }
    # Build up ancestors from children
    ancestors = calculate_ancestors(children)

    dates = {
        0: 2019,
        1: 2020,
        2: 2020,
        3: 2020,
        4: 2020,
        5: 2020,
        6: 2020,
        7: 2026,
        -1: 2010,
        -2: 2010,
        -3: 2010,
        -4: 2010,
    }

    scores = cd_index(focus, children, ancestors, time_window=5, dates=dates)

    df = pd.DataFrame(scores).set_index("focus")

    npt.assert_allclose(df.iloc[0].cd_index, 1.0)


def test_cd_against_ref_implementation():
    # dummy vertices for python module tests
    pyvertices = [
        {"name": "2Z", "time": datetime.datetime(1991, 1, 1)},
        {"name": "5Z", "time": datetime.datetime(1993, 1, 1)},
        {"name": "7Z", "time": datetime.datetime(1993, 2, 1)},
        {"name": "4Z", "time": datetime.datetime(1992, 1, 1)},
        {"name": "10Z", "time": datetime.datetime(1999, 1, 1)},
    ]

    # dummy edges for python module tests
    pyedges = [
        {"source": "4Z", "target": "2Z"},
        {"source": "5Z", "target": "2Z"},
        {"source": "7Z", "target": "5Z"},
        {"source": "10Z", "target": "4Z"},
    ]

    # create graph
    graph = cdindex.Graph()

    # add vertices
    dates = {}
    for vertex in pyvertices:
        graph.add_vertex(
            vertex["name"], cdindex.timestamp_from_datetime(vertex["time"])
        )
        dates[vertex["name"]] = vertex["time"]

    # add edges
    for edge in pyedges:
        graph.add_edge(edge["source"], edge["target"])

    TEST_TIME_PY = datetime.timedelta(days=1825 + 10)

    # Convert to the local format that is expected.
    children = collections.defaultdict(set)
    for edge in pyedges:
        children[edge["target"]].add(edge["source"])

    ancestors = calculate_ancestors(children)
    focii = [v["name"] for v in pyvertices]

    scores = cd_index(
        focii, children, ancestors, dates=dates, time_window=TEST_TIME_PY
    )
    df = pd.DataFrame(scores).set_index("focus")

    for vertex in pyvertices:
        ref_index = graph.cdindex(
            vertex["name"], int(TEST_TIME_PY.total_seconds())
        )
        calc_index = df.loc[vertex["name"]].cd_index
        print(vertex["name"], calc_index, ref_index)
        assert (
            ref_index is None and np.isnan(calc_index)
        ) or ref_index == df.loc[vertex["name"]].cd_index


def test_cd_against_random():
    graph = cdindex.RandomGraph(
        generations=(2, 3, 4, 5, 6, 7, 7, 9), edge_fraction=1
    )

    vertices = graph.vertices()
    pyvertices = []
    pyedges = []
    dates = {}
    for v in list(vertices):
        pyvertices.append(
            {
                "name": v,
                "time": datetime.datetime.fromtimestamp(graph.timestamp(v)),
            }
        )
        dates[v] = pyvertices[-1]["time"]
        out_edges = graph.out_edges(v)
        out_edges = [{"source": v, "target": e} for e in out_edges]
        pyedges.extend(out_edges)

    # Print out the connections of v6g6
    for v in pyvertices:
        if v["name"] == "v6g6":
            print(v)

    for e in pyedges:
        if e["source"] == "v6g6" or e["target"] == "v6g6":
            print(e)

    TEST_TIME_PY = datetime.timedelta(days=1825)

    # Convert to the local format that is expected.
    children = collections.defaultdict(set)
    for edge in pyedges:
        children[edge["target"]].add(edge["source"])

    ancestors = calculate_ancestors(children)
    focii = [v["name"] for v in pyvertices]

    scores = cd_index(focii, children, ancestors, TEST_TIME_PY, dates)
    df = pd.DataFrame(scores).set_index("focus")

    for vertex in pyvertices:
        ref_index = graph.cdindex(
            vertex["name"], int(TEST_TIME_PY.total_seconds())
        )
        calc_index = df.loc[vertex["name"]].cd_index
        print(vertex["name"], calc_index, ref_index)
        assert (
            ref_index is None and np.isnan(calc_index)
        ) or ref_index == df.loc[vertex["name"]].cd_index
