import datetime
from typing import Dict, List, Set

import tqdm


def protected_div(a, b):
    if b == 0:
        return None
    return a / b


def cd_index(
    focii: List[str],
    children: Dict[str, Set[str]],
    ancestors: Dict[str, Set[str]],
    time_window: datetime.timedelta,
    dates: Dict[str, datetime.datetime],
):
    # Calculate the CD index for each focal paper.
    results = []

    for focus in tqdm.tqdm(focii):
        # Find out the total forward citations for this paper.

        children_citations = {
            x
            for x in children[focus]
            if dates[x] <= dates[focus] + time_window
            and dates[x] > dates[focus]
        }
        forward_citations = children_citations.copy()
        for ancestor in ancestors[focus]:
            forward_citations |= {
                x
                for x in children[ancestor]
                if dates[x] > dates[focus]
                and dates[x] <= dates[focus] + time_window
            }

        forward_citations.discard(focus)
        forward_citations.discard(None)
        children_citations.discard(None)

        # Now look at the direct children of the focus, and for each of them,
        # find out if they cite any of the ancestors of the focus.
        plus_score = 0
        minus_score = 0
        for child in children_citations:
            if ancestors[focus].intersection(ancestors[child]):
                # Consolidating.
                minus_score += 1
            else:
                plus_score += 1
        results.append(
            {
                "focus": focus,
                "plus_score": plus_score,
                "minus_score": minus_score,
                "n_ij": len(children_citations),
                "n_k": len(forward_citations.difference(children_citations)),
                "cd_index": protected_div(
                    plus_score - minus_score, len(forward_citations)
                ),
            }
        )

    return results
