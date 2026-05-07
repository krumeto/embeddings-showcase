# Clustering Page Handoff Plan

This folder is a handoff note for implementing the Streamlit clustering page in a fresh session.

## Goal

Build an educational clustering demo that highlights EVōC's hierarchical cluster layers over the stored article embeddings.

The page should help users see that embeddings can be clustered at multiple levels of granularity:

```text
broad themes -> mid-level topics -> fine-grained clusters -> representative articles
```

## Key References

- EVōC docs: https://evoc.readthedocs.io/en/latest/
- EVōC GitHub: https://github.com/TutteInstitute/evoc
- EVōC notebook inspiration: https://molab.marimo.io/github/koaning/notebooks/blob/main/evoc-fashion.py
- Plotly treemaps: https://plotly.com/python/treemaps/
- Wigglystuff treemap reference: https://koaning.github.io/wigglystuff/reference/treemap/

## Recommendation

Use **Plotly treemaps** for the Streamlit implementation.

Reasons:

- `plotly` is already installed.
- Streamlit supports `st.plotly_chart(...)` directly.
- Plotly treemaps support click-to-zoom and pathbar navigation.
- Plotly can consume EVōC hierarchy data through `ids`, `parents`, `labels`, and `values`.
- `wigglystuff` is attractive, but it is an `anywidget` library aimed mainly at notebook-style runtimes like marimo/Jupyter/Solara. It may not work cleanly in Streamlit without extra integration work.

## Dependencies To Add

Add EVōC:

```bash
uv add evoc
```

EVōC is currently early/beta, so pinning may be wise if the install resolves cleanly:

```bash
uv add "evoc==0.3.1"
```

Verify:

```bash
uv run python - <<'PY'
import evoc
print(evoc.__version__)
PY
```

## Existing Local Artifacts

The app should use these already-generated local files:

```text
data/raw/newswire_5000_sample.parquet
data/embeddings/openai_embeddings.npy
data/embeddings/hf_embeddings.npy
```

These files are intentionally ignored by git.

## Page Location

Replace the placeholder page:

```text
app/pages/2_Clustering.py
```

Optional helper functions can live in:

```text
src/embeddings_showcase/clustering.py
```

## User Controls

Start with a small set of controls:

- Provider selector: `OpenAI` or `Hugging Face`
- OpenAI dimension selector: `384`, `512`, `1024`, `full`; default `384`
- `base_min_cluster_size`; default around `20` or `30`
- `n_neighbors`; default `15`
- `max_layers`; default `3`
- `noise_level`; default `0.5`
- `random_state`; keep fixed at `42` for reproducibility
- Button: `Run clustering`

Do not expose advanced EVōC internals in v1.

## EVōC Fit Pattern

Use sklearn-style API:

```python
from evoc import EVoC

clusterer = EVoC(
    base_min_cluster_size=base_min_cluster_size,
    n_neighbors=n_neighbors,
    max_layers=max_layers,
    noise_level=noise_level,
    random_state=42,
)
labels = clusterer.fit_predict(vectors)
```

Useful attributes:

```python
clusterer.labels_
clusterer.membership_strengths_
clusterer.cluster_layers_
clusterer.membership_strength_layers_
clusterer.cluster_tree_
clusterer.duplicates_
```

EVōC layers are ordered from fine to coarse:

```text
cluster_layers_[0] = finest layer
cluster_layers_[1] = coarser
cluster_layers_[2] = even coarser
```

For a treemap, reverse them into a root-to-leaf hierarchy:

```python
coarse = clusterer.cluster_layers_[2]
middle = clusterer.cluster_layers_[1]
fine = clusterer.cluster_layers_[0]
```

Handle fewer than 3 available layers gracefully.

## Caching

Use Streamlit caching because EVōC can take time and has Numba JIT overhead:

```python
@st.cache_data(show_spinner=False)
def load_articles(...): ...

@st.cache_data(show_spinner=False)
def load_embeddings(...): ...

@st.cache_resource(show_spinner=False)
def fit_evoc(provider, dimension_label, base_min_cluster_size, n_neighbors, max_layers, noise_level):
    ...
```

Cache keys should include all user controls that change clustering.

## Treemap Data Shape

Create a per-article dataframe after fitting:

```text
row_id
article
known labels
layer_0 / fine
layer_1 / medium
layer_2 / coarse
membership_strength
```

Then create treemap paths. Example:

```text
root = "All articles"
coarse = "Layer 2 / Cluster 4"
medium = "Layer 1 / Cluster 11"
fine = "Layer 0 / Cluster 38"
```

Treat noise label `-1` as `"Noise"` at each level.

Aggregate:

```python
tree_df = (
    article_cluster_df
    .groupby(["root", "coarse", "medium", "fine"], dropna=False)
    .agg(
        count=("row_id", "size"),
        avg_strength=("membership_strength", "mean"),
        dominant_label=(..., ...)
    )
    .reset_index()
)
```

Simpler v1: aggregate count only, then compute label summaries separately for hover text.

## Plotly Treemap

Start with `plotly.express`:

```python
import plotly.express as px

fig = px.treemap(
    tree_df,
    path=["root", "coarse", "medium", "fine"],
    values="count",
    color="dominant_label",
    hover_data=["count", "avg_strength"],
)
fig.update_traces(root_color="lightgrey")
fig.update_layout(margin=dict(t=20, l=10, r=10, b=10))
st.plotly_chart(fig, width="stretch")
```

If explicit cluster ids/parents are needed, switch to `plotly.graph_objects.go.Treemap`.

Plotly gives zoom-in/out interaction through rectangle clicks and the pathbar.

## Detail Pane

Plotly treemap click selection may not reliably expose clicked treemap nodes in Streamlit the same way scatter selections do. For v1, use a separate selector:

- Dropdown: choose layer
- Dropdown: choose cluster id within that layer

Show:

- cluster size
- known label distribution
- average membership strength
- 5-10 representative article snippets

Representatives can be chosen by highest membership strength within the selected cluster.

If we later get treemap click events working well, wire clicked cluster path to the detail pane.

## Educational Copy To Include

Explain EVōC briefly:

> EVōC is designed for clustering high-dimensional embedding vectors. It builds a neighbor graph, learns a cluster-friendly node embedding, and extracts density-based clusters at multiple granularities.

Explain the hierarchy:

> The outer rectangles show broad clusters. Clicking inward reveals progressively finer clusters. This is useful for topic exploration because news articles often have structure at several resolutions.

Explain noise:

> Points labelled `-1` are treated as noise: articles that do not fit confidently into a cluster at that layer.

## Suggested Acceptance Criteria

- `uv add evoc` succeeds and `uv.lock` is updated.
- `app/pages/2_Clustering.py` loads without errors.
- User can choose OpenAI or HF embeddings.
- User can run EVōC clustering and see:
  - number of clusters
  - number/proportion of noise points
  - number of hierarchy layers
  - interactive Plotly treemap
  - representative articles for a selected layer/cluster
- Runs are cached across reruns for the same settings.
- `uv run python -m compileall app src` passes.

## Nice Follow-Ups

- Compare OpenAI vs HF cluster counts side by side.
- Color treemap by dominant known label or cluster purity.
- Add UMAP/2D scatter colored by selected EVōC layer.
- Add duplicate/near-duplicate inspection from `clusterer.duplicates_`.
- Persist cluster results to ignored local parquet/npy artifacts to avoid re-fitting after restart.
