# Role Mining Project

This project performs role mining from entitlement data using Apriori (frequent itemset mining) and clustering (KMeans / DBSCAN).
It provides a Streamlit UI to run the pipeline, tune parameters, and infer suggested roles for a given set of entitlements.

## Files
- `role_mining/data_pipeline.py` - data loading and preprocessing (drops role_id).
- `role_mining/mining_models.py` - apriori, clustering, candidate role generation and matching.
- `ui/role_mining_app.py` - Streamlit UI to run pipeline and perform inference.
- `data/entitlements.csv` - your uploaded CSV (copied from the provided file).
- `requirements.txt` - Python dependencies.

## Run locally
1. Create virtualenv and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. Run Streamlit UI:
   ```bash
   streamlit run ui/role_mining_app.py
   ```

## Notes
- The data loader expects an `entitlements` column (comma-separated strings or JSON-like list).
- The UI exposes sliders for apriori support, minimum popularity, coverage fraction, clustering params, and test split.
- Candidate roles are stored in `st.session_state["candidates"]` after pipeline runs.