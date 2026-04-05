"""CrossCat model initialization and joint probability.

Maps to the original State.h / State.cpp from probcomp/crosscat.
The original State class managed initialization from prior, transition
dispatching, and state serialization. Here we separate concerns:
- model.py: initialization and scoring
- gibbs.py: transition kernels
- types.py: state representation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln

from crosscat.components import (
    BetaBernoulli,
    DirichletCategorical,
    NormalGamma,
    OrderedLogistic,
    VonMises,
)
from crosscat.types import (
    LOG_EPS,
    ColumnHypers,
    ColumnType,
    CrossCatState,
    SufficientStats,
    ViewState,
)


def _crp_sample(rng_key: Array, alpha: float, n: int) -> Array:
    """Sample assignments from a Chinese Restaurant Process.

    Sequential CRP: each item sits at an existing table with probability
    proportional to occupancy, or starts a new table with probability
    proportional to alpha.

    Args:
        rng_key: JAX PRNG key.
        alpha: CRP concentration parameter.
        n: Number of items to assign.

    Returns:
        Integer assignments array of shape (n,).
    """

    def _step(carry, key):
        assignments, counts, n_tables = carry
        # Probabilities: existing tables proportional to counts, new table proportional to alpha
        max_tables = n  # upper bound
        table_probs = jnp.where(jnp.arange(max_tables) < n_tables, counts[:max_tables], 0.0)
        # Probability of new table
        new_table_probs = table_probs.at[n_tables].set(alpha)
        log_probs = jnp.log(new_table_probs + LOG_EPS)
        chosen = jax.random.categorical(key, log_probs)

        # Update
        is_new = chosen == n_tables
        new_n_tables = jnp.where(is_new, n_tables + 1, n_tables)
        new_counts = counts.at[chosen].add(1.0)
        return (assignments, new_counts, new_n_tables), chosen

    keys = jax.random.split(rng_key, n)
    init_counts = jnp.zeros(n, dtype=jnp.float32)
    init_state = (jnp.zeros(n, dtype=jnp.int32), init_counts, jnp.array(0, dtype=jnp.int32))

    _, assignments = jax.lax.scan(_step, init_state, keys)
    return assignments


def _safe_n_categories(col_data: Array) -> int:
    """Compute number of categories from column data, handling all-NaN gracefully."""
    clean = col_data[~jnp.isnan(col_data)]
    if clean.shape[0] == 0:
        return 2  # default minimum for empty/all-NaN columns
    return max(int(jnp.max(clean)) + 1, 1)


def _default_hypers(column_type: ColumnType, col_data: Array) -> ColumnHypers:
    """Initialize column hyperparameters from data-driven defaults.

    Following the original CrossCat initialization strategy:
    - Continuous: set mu to data mean, s to data variance, r=1, nu=2
    - Categorical: alpha = 1 (symmetric)
    - Binary: alpha=1, beta=1 (uniform prior)
    - Ordinal: cutpoints as linspace (but using Dirichlet internally)
    """
    if column_type == ColumnType.CONTINUOUS:
        clean = col_data[~jnp.isnan(col_data)]
        mean = jnp.nanmean(col_data) if clean.shape[0] > 0 else jnp.array(0.0)
        var = jnp.nanvar(col_data) + 1e-6 if clean.shape[0] > 0 else jnp.array(1.0)
        return ColumnHypers(
            column_type=column_type,
            mu=mean,
            r=jnp.array(1.0),
            s=var,
            nu=jnp.array(2.0),
        )
    elif column_type == ColumnType.CATEGORICAL:
        return ColumnHypers(
            column_type=column_type,
            dirichlet_alpha=jnp.array(1.0),
        )
    elif column_type == ColumnType.BINARY:
        return ColumnHypers(
            column_type=column_type,
            alpha=jnp.array(1.0),
            beta=jnp.array(1.0),
        )
    elif column_type == ColumnType.ORDINAL:
        n_levels = _safe_n_categories(col_data)
        return ColumnHypers(
            column_type=column_type,
            mu=jnp.array(0.0),
            s=jnp.array(4.0),
            cutpoints=jnp.linspace(-2.0, 2.0, max(n_levels - 1, 1)),
        )
    elif column_type == ColumnType.CYCLIC:
        return ColumnHypers(
            column_type=column_type,
            kappa=jnp.array(1.0),  # likelihood concentration
            vm_a=jnp.array(1.0),  # prior concentration on mean direction
            vm_mu=jnp.array(jnp.pi),  # prior mean direction (b)
        )
    else:
        raise ValueError(f"Unknown column type: {column_type}")


def _compute_suffstats_for_view(
    data: Array,
    column_indices: Array,
    column_types: list[ColumnType],
    row_assignments: Array,
    n_clusters: int,
) -> list[list[SufficientStats]]:
    """Compute sufficient statistics for each (cluster, column) in a view.

    Delegates to ``compute_suffstats_vectorized`` (the canonical vectorized
    implementation) and converts the packed arrays back to dataclass lists.

    Returns:
        Nested list: suffstats[cluster_idx][col_idx_in_view]
    """
    from crosscat.packed.state import _TYPE_TO_ID
    from crosscat.packed.suffstats import compute_suffstats_vectorized

    col_indices_arr = jnp.asarray(column_indices, dtype=jnp.int32)
    col_type_ids = jnp.array([_TYPE_TO_ID[ct] for ct in column_types], dtype=jnp.int32)

    # Only compute max_categories from categorical/ordinal columns to avoid
    # massive one-hot arrays from continuous column values.
    cat_ord_cols = [
        int(c)
        for c in column_indices
        if column_types[int(c)] in (ColumnType.CATEGORICAL, ColumnType.ORDINAL)
    ]
    if cat_ord_cols:
        max_categories = max(_safe_n_categories(data[:, c]) for c in cat_ord_cols)
    else:
        max_categories = 2
    max_categories = max(max_categories, 2)

    counts, sum_x, sum_x_sq, cat_counts, sum_sin, sum_cos = compute_suffstats_vectorized(
        data,
        col_indices_arr,
        col_type_ids,
        jnp.asarray(row_assignments, dtype=jnp.int32),
        n_clusters,
        max_clusters=n_clusters,
        max_categories=max_categories,
    )

    # Convert packed arrays back to nested SufficientStats dataclass lists
    all_suffstats = []
    for c in range(n_clusters):
        cluster_stats = []
        for local_idx in range(len(column_indices)):
            col_idx = int(column_indices[local_idx])
            col_type = column_types[col_idx]
            count = counts[c, local_idx]

            if col_type == ColumnType.CONTINUOUS:
                ss = SufficientStats(
                    column_type=col_type,
                    count=count,
                    sum_x=sum_x[c, local_idx],
                    sum_x_sq=sum_x_sq[c, local_idx],
                )
            elif col_type in (ColumnType.CATEGORICAL, ColumnType.ORDINAL):
                ss = SufficientStats(
                    column_type=col_type,
                    count=count,
                    category_counts=cat_counts[c, local_idx],
                )
            elif col_type == ColumnType.BINARY:
                ss = SufficientStats(
                    column_type=col_type,
                    count=count,
                    sum_x=sum_x[c, local_idx],
                )
            elif col_type == ColumnType.CYCLIC:
                ss = SufficientStats(
                    column_type=col_type,
                    count=count,
                    sum_sin=sum_sin[c, local_idx],
                    sum_cos=sum_cos[c, local_idx],
                )
            else:
                raise ValueError(f"Unknown column type: {col_type}")
            cluster_stats.append(ss)
        all_suffstats.append(cluster_stats)
    return all_suffstats


def initialize(
    rng_key: Array,
    data: Array,
    column_types: list[ColumnType],
    *,
    n_chains: int = 1,
    column_crp_alpha: float = 1.0,
    row_crp_alpha: float = 1.0,
    initialization: str = "from_the_prior",
    subsample_rows: int | None = None,
) -> CrossCatState | list[CrossCatState]:
    """Initialize CrossCat state(s).

    Maps to original LocalEngine.initialize() which calls State.__init__
    with initialization='from_the_prior'.

    Args:
        rng_key: JAX PRNG key.
        data: Observation matrix, shape (n_rows, n_cols).
        column_types: Type specification per column.
        n_chains: Number of independent chains to initialize.
        column_crp_alpha: Initial outer DP concentration.
        row_crp_alpha: Initial inner DP concentration per view.
        initialization: One of 'from_the_prior', 'together', 'apart'.
            'from_the_prior': Sample assignments from CRP (default).
            'together': All columns in one view.
            'apart': Each column in its own view.
        subsample_rows: If set, initialize on a random subsample of this many
            rows. Hyperparameters are computed from the full data for accurate
            priors, but CRP row assignments and sufficient statistics use only
            the subsample. The returned state has n_rows=subsample_rows.
            Remaining rows can be streamed in via ``packed_insert_rows``.

    Returns:
        Single CrossCatState if n_chains=1, else list of states.
        When subsample_rows is set, also returns the subsample indices as a
        second element: ``(state, subsample_idx)`` or
        ``(states_list, subsample_idx)``.

    Raises:
        ValueError: If data is empty, column_types length mismatches data, or
            invalid initialization mode.
    """
    if data.ndim != 2:
        raise ValueError(f"Data must be 2-dimensional, got shape {data.shape}")
    n_rows, n_cols = data.shape
    if n_rows == 0:
        raise ValueError("Data must have at least one row")
    if n_cols == 0:
        raise ValueError("Data must have at least one column")
    if len(column_types) != n_cols:
        raise ValueError(
            f"column_types length ({len(column_types)}) must match number of columns ({n_cols})"
        )
    valid_inits = {"from_the_prior", "together", "apart"}
    if initialization not in valid_inits:
        raise ValueError(
            f"Unknown initialization '{initialization}'. Must be one of {valid_inits}"
        )

    # Handle subsampling
    subsample_idx = None
    if subsample_rows is not None:
        if subsample_rows >= n_rows:
            subsample_rows = None  # No subsampling needed
        elif subsample_rows < 1:
            raise ValueError(f"subsample_rows must be >= 1, got {subsample_rows}")
        else:
            rng_key, sub_key = jax.random.split(rng_key)
            subsample_idx = jax.random.choice(
                sub_key, n_rows, shape=(subsample_rows,), replace=False
            )
            subsample_idx = jnp.sort(subsample_idx)

    # Data for suffstats: subsample if requested, full otherwise
    init_data = data[subsample_idx] if subsample_idx is not None else data
    init_n_rows = init_data.shape[0]

    def _init_one(key):
        k1, k2 = jax.random.split(key)

        # Step 1: Column-to-view assignments based on initialization mode
        if initialization == "together":
            col_assignments = jnp.zeros(n_cols, dtype=jnp.int32)
        elif initialization == "apart":
            col_assignments = jnp.arange(n_cols, dtype=jnp.int32)
        else:
            # from_the_prior: sample from CRP
            col_assignments = _crp_sample(k1, column_crp_alpha, n_cols)

        n_views = int(jnp.max(col_assignments)) + 1

        # Step 2: Initialize column hyperparameters from FULL data
        col_hypers = []
        for j in range(n_cols):
            col_hypers.append(_default_hypers(column_types[j], data[:, j]))

        # Step 3: For each view, sample row assignments and compute suffstats
        # Uses init_data (subsample) for assignments and suffstats
        view_keys = jax.random.split(k2, n_views)
        views = []
        for v in range(n_views):
            col_mask = col_assignments == v
            col_indices = jnp.arange(n_cols)[col_mask]

            # Sample row-to-cluster assignments for this view
            row_assigns = _crp_sample(view_keys[v], row_crp_alpha, init_n_rows)
            n_clusters = int(jnp.max(row_assigns)) + 1

            # Compute sufficient statistics on init_data
            suffstats = _compute_suffstats_for_view(
                init_data, col_indices, column_types, row_assigns, n_clusters
            )

            views.append(
                ViewState(
                    column_indices=col_indices,
                    row_assignments=row_assigns,
                    row_crp_alpha=jnp.array(row_crp_alpha),
                    suffstats=suffstats,
                )
            )

        return CrossCatState(
            column_assignments=col_assignments,
            column_crp_alpha=jnp.array(column_crp_alpha),
            column_hypers=col_hypers,
            column_types=column_types,
            views=views,
            n_rows=init_n_rows,
            n_cols=n_cols,
        )

    if n_chains == 1:
        result = _init_one(rng_key)
    else:
        keys = jax.random.split(rng_key, n_chains)
        result = [_init_one(keys[i]) for i in range(n_chains)]

    if subsample_idx is not None:
        return result, subsample_idx
    return result


def insert_rows(
    rng_key: Array,
    state: CrossCatState,
    data: Array,
    new_rows: Array,
) -> tuple[CrossCatState, Array]:
    """Insert new rows into an existing CrossCat state.

    Maps to original LocalEngine.insert(). New rows are assigned to clusters
    via the CRP predictive distribution (no re-inference on existing rows).

    Args:
        rng_key: JAX PRNG key.
        state: Current CrossCat state.
        data: Original observation matrix, shape (n_rows, n_cols).
        new_rows: New observations, shape (n_new, n_cols).

    Returns:
        Tuple of (updated_state, updated_data) with new rows incorporated.
    """
    n_new = new_rows.shape[0]
    updated_data = jnp.concatenate([data, new_rows], axis=0)
    keys = jax.random.split(rng_key, len(state.views))

    new_views = []
    for v_idx, view in enumerate(state.views):
        row_keys = jax.random.split(keys[v_idx], n_new)
        row_assignments = view.row_assignments
        n_clusters = int(jnp.max(row_assignments)) + 1
        alpha = view.row_crp_alpha
        n_existing_clusters = n_clusters

        new_assigns = []
        for i in range(n_new):
            # CRP predictive: score each existing cluster + new cluster
            cluster_counts = jnp.array(
                [jnp.sum(row_assignments == c) for c in range(n_existing_clusters)]
            ).astype(jnp.float32)
            log_probs = jnp.log(cluster_counts + LOG_EPS)
            # Add likelihood of row under each cluster using existing suffstats
            row_data = new_rows[i]
            for c in range(n_existing_clusters):
                for local_idx, col_idx_val in enumerate(view.column_indices.tolist()):
                    col_idx = int(col_idx_val)
                    col_type = state.column_types[col_idx]
                    hypers = state.column_hypers[col_idx]
                    ss = view.suffstats[c][local_idx]
                    comp = _get_component_class(col_type)
                    x = row_data[col_idx]
                    if not jnp.isnan(x):
                        log_lik = comp.posterior_predictive_logp(x, ss, hypers)
                        log_probs = log_probs.at[c].add(log_lik)

            # New cluster — use CRP prior only (empty cluster prior predictive)
            log_new = jnp.log(alpha)
            log_probs = jnp.concatenate([log_probs, jnp.array([log_new])])
            log_probs = log_probs - jnp.max(log_probs)
            chosen = int(jax.random.categorical(row_keys[i], log_probs))

            if chosen >= n_existing_clusters:
                chosen = n_clusters
                n_clusters += 1
            new_assigns.append(chosen)

        # Extend row assignments
        new_row_assigns = jnp.concatenate(
            [row_assignments, jnp.array(new_assigns, dtype=jnp.int32)]
        )

        # Recompute suffstats with extended data
        n_clusters_final = int(jnp.max(new_row_assigns)) + 1
        suffstats = _compute_suffstats_for_view(
            updated_data,
            view.column_indices,
            state.column_types,
            new_row_assigns,
            n_clusters_final,
        )

        new_views.append(
            ViewState(
                column_indices=view.column_indices,
                row_assignments=new_row_assigns,
                row_crp_alpha=view.row_crp_alpha,
                suffstats=suffstats,
            )
        )

    new_state = CrossCatState(
        column_assignments=state.column_assignments,
        column_crp_alpha=state.column_crp_alpha,
        column_hypers=state.column_hypers,
        column_types=state.column_types,
        views=new_views,
        n_rows=state.n_rows + n_new,
        n_cols=state.n_cols,
    )
    return new_state, updated_data


def _get_component_class(col_type: ColumnType):
    """Return component model class for dispatch."""
    if col_type == ColumnType.CONTINUOUS:
        return NormalGamma
    elif col_type == ColumnType.CATEGORICAL:
        return DirichletCategorical
    elif col_type == ColumnType.BINARY:
        return BetaBernoulli
    elif col_type == ColumnType.ORDINAL:
        return OrderedLogistic
    elif col_type == ColumnType.CYCLIC:
        return VonMises
    else:
        raise ValueError(f"Unknown column type: {col_type}")


def _log_crp(assignments: Array, alpha: Array) -> Array:
    """Log probability of a partition under a CRP.

    log p(z | alpha) = K * log(alpha) + sum_k gammaln(n_k) - gammaln(N + alpha) + gammaln(alpha)

    where K = number of clusters, n_k = size of cluster k, N = total items.
    """
    n = assignments.shape[0]
    n_clusters = int(jnp.max(assignments)) + 1
    counts = jnp.bincount(assignments, length=n_clusters).astype(jnp.float32)

    log_p = (
        n_clusters * jnp.log(alpha)
        + jnp.sum(gammaln(counts))
        - gammaln(n + alpha)
        + gammaln(alpha)
    )
    return log_p


def log_joint(state: CrossCatState, data: Array) -> Array:
    """Compute the joint log probability of state and data.

    log p(state, data) = log p(column_partition | alpha_outer)
                       + sum_v log p(row_partition_v | alpha_v)
                       + sum_v sum_c sum_col log p(data[rows_c, col] | hypers_col)
                       + log p(alpha_outer) + sum_v log p(alpha_v)
                       + sum_col log p(hypers_col)

    Maps to scoring logic spread across original State.cpp, View.cpp,
    and numerics.cpp (calc_crp_logp, calc_continuous_logp, etc.).

    Args:
        state: Current CrossCat state.
        data: Observation matrix, shape (n_rows, n_cols).

    Returns:
        Scalar log joint probability.
    """
    log_p = jnp.array(0.0)

    # 1. Log CRP probability for column partition
    log_p = log_p + _log_crp(state.column_assignments, state.column_crp_alpha)

    # 2. For each view: log CRP for row partition + data likelihood
    for view in state.views:
        # Row partition CRP
        log_p = log_p + _log_crp(view.row_assignments, view.row_crp_alpha)

        # Data likelihood for each (cluster, column) pair
        if view.suffstats is not None:
            n_clusters = len(view.suffstats)
            for c in range(n_clusters):
                for local_idx, col_idx in enumerate(view.column_indices.tolist()):
                    col_idx = int(col_idx)
                    hypers = state.column_hypers[col_idx]
                    ss = view.suffstats[c][local_idx]
                    col_type = state.column_types[col_idx]

                    comp = _get_component_class(col_type)
                    log_p = log_p + comp.log_marginal_likelihood(ss, hypers)

    # 3. Gamma(1,1) prior on CRP alphas
    log_p = log_p - state.column_crp_alpha  # Exp(1) prior
    for view in state.views:
        log_p = log_p - view.row_crp_alpha

    return log_p
