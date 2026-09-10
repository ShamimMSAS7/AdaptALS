
"""
Pareto-based optimizer for AdaptALS.

Objectives:
    1. Maximize validation NDCG@10
    2. Minimize the number of physical ALS retrainings

Procedure:
    1. Remove dominated configurations.
    2. Construct the Pareto frontier.
    3. Normalize NDCG and retraining count.
    4. Select the knee point of the frontier.

Each candidate must contain:
    {
        "gamma": float,
        "lambda": float,
        "tau": float,
        "ndcg": float,
        "retrains": int
    }
"""


def dominates(a, b):
    """
    Return True if candidate a dominates candidate b.

    NDCG is maximized.
    Retraining count is minimized.

    Candidate a dominates b when:
        a.ndcg >= b.ndcg
        a.retrains <= b.retrains

    and at least one of these inequalities is strict.
    """

    ndcg_better_or_equal = a["ndcg"] >= b["ndcg"]
    retrains_better_or_equal = a["retrains"] <= b["retrains"]

    at_least_one_strict = (
        a["ndcg"] > b["ndcg"]
        or a["retrains"] < b["retrains"]
    )

    return (
        ndcg_better_or_equal
        and retrains_better_or_equal
        and at_least_one_strict
    )


def get_pareto_front(candidates):
    """
    Return the non-dominated Pareto frontier.

    Parameters
    ----------
    candidates : list of dict
        Validation results for all parameter configurations.

    Returns
    -------
    list of dict
        Non-dominated configurations.
    """

    if not candidates:
        return []

    pareto_front = []

    for candidate in candidates:

        dominated = False

        for other in candidates:

            if other is candidate:
                continue

            if dominates(other, candidate):
                dominated = True
                break

        if not dominated:
            pareto_front.append(candidate)

    return pareto_front


def normalize_objectives(pareto_front):
    """
    Normalize Pareto-front objectives to [0, 1].

    NDCG:
        higher is better

    Retraining count:
        lower is better

    Returns
    -------
    list of dict
        Pareto configurations with normalized objectives.
    """

    if not pareto_front:
        return []

    ndcg_values = [x["ndcg"] for x in pareto_front]
    retrain_values = [x["retrains"] for x in pareto_front]

    ndcg_min = min(ndcg_values)
    ndcg_max = max(ndcg_values)

    retrain_min = min(retrain_values)
    retrain_max = max(retrain_values)

    normalized = []

    for candidate in pareto_front:

        # NDCG: maximize
        if ndcg_max == ndcg_min:
            ndcg_norm = 1.0
        else:
            ndcg_norm = (
                (candidate["ndcg"] - ndcg_min)
                / (ndcg_max - ndcg_min)
            )

        # Retraining: minimize
        # Convert it so that larger value means better.
        if retrain_max == retrain_min:
            retrain_norm = 1.0
        else:
            retrain_norm = (
                (retrain_max - candidate["retrains"])
                / (retrain_max - retrain_min)
            )

        item = candidate.copy()

        item["ndcg_norm"] = ndcg_norm
        item["retrain_norm"] = retrain_norm

        normalized.append(item)

    return normalized


def select_knee_point(pareto_front):
    """
    Select the knee point of a normalized Pareto frontier.

    The ideal point is:
        (1, 1)

    where:
        NDCG = maximum
        Retraining efficiency = maximum

    The knee point is selected as the Pareto configuration
    closest to the ideal point.

    Returns
    -------
    dict or None
        Selected configuration.
    """

    normalized = normalize_objectives(pareto_front)

    if not normalized:
        return None

    best = None
    best_distance = float("inf")

    for candidate in normalized:

        distance = (
            (1.0 - candidate["ndcg_norm"]) ** 2
            + (1.0 - candidate["retrain_norm"]) ** 2
        ) ** 0.5

        candidate["ideal_distance"] = distance

        if distance < best_distance:
            best_distance = distance
            best = candidate

    return best


def select_pareto_knee(candidates):
    """
    Complete Pareto optimization procedure.

    Steps:
        1. Find non-dominated configurations.
        2. Construct the Pareto frontier.
        3. Normalize the objectives.
        4. Select the knee point.

    Parameters
    ----------
    candidates : list of dict

    Returns
    -------
    dict
        Selected parameter configuration.
    """

    if not candidates:
        return None

    pareto_front = get_pareto_front(candidates)

    selected = select_knee_point(pareto_front)

    return selected


def print_pareto_results(candidates, selected=None):
    """
    Print the complete optimization result for analysis.
    """

    pareto_front = get_pareto_front(candidates)

    print("\n" + "=" * 70)
    print("PARETO OPTIMIZATION RESULTS")
    print("=" * 70)

    print(f"\nTotal configurations: {len(candidates)}")
    print(f"Pareto-optimal configurations: {len(pareto_front)}")

    print("\nPareto Frontier:")
    print("-" * 70)

    # Sort by retraining count.
    frontier_sorted = sorted(
        pareto_front,
        key=lambda x: (x["retrains"], -x["ndcg"])
    )

    for i, candidate in enumerate(frontier_sorted, 1):

        print(
            f"{i:2d}. "
            f"gamma={candidate['gamma']:.2f}, "
            f"lambda={candidate['lambda']:.2f}, "
            f"tau={candidate['tau']:.2f} | "
            f"NDCG@10={candidate['ndcg']:.6f} | "
            f"Retrains={candidate['retrains']}"
        )

    if selected is not None:

        print("\n" + "-" * 70)
        print("SELECTED KNEE POINT")
        print("-" * 70)

        print(
            f"gamma  = {selected['gamma']:.2f}\n"
            f"lambda = {selected['lambda']:.2f}\n"
            f"tau    = {selected['tau']:.2f}\n"
            f"NDCG@10 = {selected['ndcg']:.6f}\n"
            f"Retrains = {selected['retrains']}\n"
            f"Normalized NDCG = {selected['ndcg_norm']:.6f}\n"
            f"Normalized retraining efficiency = "
            f"{selected['retrain_norm']:.6f}\n"
            f"Distance from ideal point = "
            f"{selected['ideal_distance']:.6f}"
        )

    print("=" * 70)
