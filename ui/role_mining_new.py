# ui/role_mining_filtered_app.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd
import math
import ast
from collections import Counter
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import numpy as np

# Import pipeline helpers from your role_mining package.
# Make sure role_mining package (data_pipeline.py, mining_models.py) is present and importable.
from role_mining.data_pipeline import load_csv, build_user_item_matrix, train_test_split_matrix
from role_mining.mining_models import (
    apriori_candidate_roles, cluster_roles,
    suggest_roles_from_apriori, suggest_roles_from_clusters,
    dedupe_candidates, match_roles_to_input, assign_roles_to_users, evaluate_candidate_set
)

st.set_page_config(page_title='Role Mining (Filtered)', layout='wide')
st.title('🔎 Role Mining — Filtered by provided entitlements (No LLM)')

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("🔧 Controls - Filter & Pipeline")

DATA_PATH = st.sidebar.text_input(
    "CSV file path",
    value=str(ROOT / "data" / "entitlements_ready.csv"),
    help="CSV must include an 'All_Entitlements' column (comma-separated)."
)

input_ents_raw = st.sidebar.text_area(
    "Enter entitlements (comma-separated). Only users who have any of these entitlements will be considered.",
    value="ENT001, ENT002",
    height=80
)
st.sidebar.caption("Leave blank to run pipeline on the full dataset (original behavior).")

# Discovery / tuning
MIN_SUPPORT = st.sidebar.slider("Apriori min support", 0.001, 0.5, 0.01, 0.001)
MIN_POPULARITY = st.sidebar.slider("Min role popularity (train filter)", 1, 500, 10, 1)
ROLE_MIN_SHARE = st.sidebar.slider("Role min share of total entitlements (fraction)", 0.0, 1.0, 0.5, 0.05)
ROLE_ASSIGN_FRAC = st.sidebar.slider("Assignment fraction (user must have >= this of role entitlements)", 0.1, 1.0, 1.0, 0.05)

CLUSTER_METHOD = st.sidebar.selectbox("Clustering method", ["kmeans", "dbscan"])
KMEANS_N = st.sidebar.slider("KMeans n_clusters", 2, 40, 5)
DBSCAN_EPS = st.sidebar.slider("DBSCAN eps (DBSCAN only)", 0.05, 1.0, 0.5, 0.01)

TEST_SIZE = st.sidebar.slider("Test size (hold-out fraction)", 0.05, 0.5, 0.2, 0.05)

# Inference parameters
MIN_COVERAGE = st.sidebar.slider("Inference coverage — fraction of provided entitlements role must cover", 0.1, 1.0, 0.5, 0.05)
MIN_POP_INFER = st.sidebar.slider("Inference min popularity (final filter)", 1, 500, 10, 1)
MODE = st.sidebar.selectbox("Matching mode (inference)", ["Strict subset (no extras allowed)", "Flexible (allow role extras)"])
MAX_EXTRA = 0
if MODE.startswith("Flexible"):
    MAX_EXTRA = st.sidebar.slider("Max extra entitlements allowed in suggested role (Flexible mode)", 0, 10, 1)

# Optional fuzzy mapping helpers
st.sidebar.markdown("---")
fuzzy_enabled = st.sidebar.checkbox("Enable fuzzy mapping of input tokens to dataset vocabulary", value=True)
fuzzy_threshold = st.sidebar.slider("Fuzzy match cutoff (0-1)", 0.5, 1.0, 0.80, 0.01)

# ---------------------------
# Utilities: normalize & parse All_Entitlements cell
# ---------------------------
def normalize_token(t):
    if t is None:
        return ""
    return str(t).strip().upper()

def parse_all_entitlements_cell(val):
    """Parse the All_Entitlements CSV cell which is usually a comma-separated string
       like "ENT001, ENT002, ENT003". Returns a list of normalized tokens."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        return [normalize_token(x) for x in val if str(x).strip()]
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return []
    # If it's a python-list-like string
    if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple, set)):
                return [normalize_token(x) for x in obj if str(x).strip()]
        except Exception:
            pass
    # Split on comma (this matches your CSV)
    if ',' in s:
        parts = [p.strip() for p in s.split(',') if p.strip()]
        return [normalize_token(x) for x in parts]
    # Single token fallback
    return [normalize_token(s)]

def ensure_user_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a user_id column. Prefer EmpID -> user_id -> Email -> EmailAddress.
       Otherwise create synthetic U0001.. IDs using the index."""
    df = df.copy()
    if 'EmpID' in df.columns:
        df['user_id'] = df['EmpID'].astype(str)
        return df
    if 'user_id' in df.columns:
        df['user_id'] = df['user_id'].astype(str)
        return df
    for c in ('Email', 'email', 'EmailAddress', 'Email_Address'):
        if c in df.columns:
            df['user_id'] = df[c].astype(str)
            return df
    df = df.reset_index(drop=False).rename(columns={'index': 'idx_for_userid'})
    df['user_id'] = df['idx_for_userid'].apply(lambda i: f"U{i+1:05d}")
    df.drop(columns=['idx_for_userid'], inplace=True)
    return df

# ---------------------------
# Run pipeline (FILTERED BY INPUT_ENTS)
# ---------------------------
if st.button("Run pipeline (filtered by provided entitlements)"):
    # Load CSV
    try:
        df_raw = pd.read_csv(DATA_PATH, dtype=str)
    except Exception as e:
        st.error(f"Could not open CSV at {DATA_PATH}: {e}")
        st.stop()

    st.success(f"Loaded raw CSV: {len(df_raw)} rows")

    # Ensure All_Entitlements exists (or detect column)
    if 'All_Entitlements' not in df_raw.columns:
        st.warning("'All_Entitlements' column not found. Attempting detection.")
        candidate_col = None
        for c in df_raw.columns:
            try:
                if df_raw[c].astype(str).str.contains('ENT').any() and df_raw[c].astype(str).str.contains(',').any():
                    candidate_col = c
                    break
            except Exception:
                continue
        if candidate_col:
            st.info(f"Using detected column '{candidate_col}' as entitlement source.")
            df_raw['All_Entitlements'] = df_raw[candidate_col].astype(str)
        else:
            st.error("No entitlement-like column found. Make sure your CSV has 'All_Entitlements' column.")
            st.stop()

    # Parse entitlements into list column
    df_raw['__ent_list_normalized'] = df_raw['All_Entitlements'].fillna("").apply(parse_all_entitlements_cell)
    # Provide alias expected by pipeline
    df_raw['ent_list'] = df_raw['__ent_list_normalized']

    # Ensure stable user_id exists
    df_raw = ensure_user_id_column(df_raw)

    # Debug preview
    with st.expander("Preview parsed entitlements (first 12 rows)"):
        preview = df_raw[['user_id', 'All_Entitlements', '__ent_list_normalized']].head(12).copy()
        preview['__ent_list_normalized'] = preview['__ent_list_normalized'].apply(lambda lst: ", ".join(lst))
        st.dataframe(preview, use_container_width=True)

    # Build vocabulary
    vocab = sorted({tok for lst in df_raw['__ent_list_normalized'].tolist() for tok in lst if tok})
    st.sidebar.write(f"Total distinct entitlements (vocabulary): {len(vocab)}")

    # Normalize user input tokens
    raw_items = [t for t in [x.strip() for x in input_ents_raw.split(",")] if t.strip()]
    input_tokens_norm = [normalize_token(x) for x in raw_items]
    input_tokens_norm = list(dict.fromkeys(input_tokens_norm))
    st.write(f"Normalized input tokens: {input_tokens_norm}")

    # Filtering logic (if user provided tokens)
    if not input_tokens_norm:
        st.info("No input entitlements supplied — running pipeline on full dataset.")
        df_filtered = df_raw.copy()
        chosen_set = set()
    else:
        # Map input tokens to vocab (exact + fuzzy)
        mapping_debug = []
        for tok in input_tokens_norm:
            if tok in vocab:
                mapping_debug.append((tok, tok, 1.0, "exact"))
                continue
            best = None
            best_score = 0.0
            for v in vocab:
                s = SequenceMatcher(None, tok, v).ratio()
                if s > best_score:
                    best_score = s
                    best = v
            mapping_debug.append((tok, best if best_score > 0 else None, round(best_score, 3),
                                  "fuzzy" if best_score > 0 else "no_match"))

        with st.expander("Mapping debug (input → vocab)"):
            st.dataframe(pd.DataFrame(mapping_debug, columns=["input", "mapped", "score", "type"]), use_container_width=True)

        auto_choices = [m[1] for m in mapping_debug if m[1] and (m[2] >= fuzzy_threshold or m[2] == 1.0)]
        auto_choices = list(dict.fromkeys([normalize_token(x) for x in auto_choices if x]))
        chosen = st.multiselect("Mapped tokens to use for filtering (edit if needed):", options=vocab, default=auto_choices)

        if not chosen:
            st.warning("No mapped tokens selected. Running pipeline on full dataset.")
            df_filtered = df_raw.copy()
            chosen_set = set()
        else:
            chosen_set = set(normalize_token(t) for t in chosen)
            def row_has_any(row):
                ents = row.get('ent_list') or row.get('__ent_list_normalized') or []
                return len(set(ents) & chosen_set) > 0
            df_filtered = df_raw[df_raw.apply(row_has_any, axis=1)].copy()
            st.success(f"Matched users: {len(df_filtered)} for tokens {sorted(list(chosen_set))}")

    if df_filtered.empty:
        st.error("No users matched the filter. Inspect mapping preview and adjust tokens or dataset formatting.")
        st.stop()

    # Save filtered raw DF for later
    st.session_state['filtered_raw_df'] = df_filtered

    # Ensure compatibility
    if 'ent_list' not in df_filtered.columns and '__ent_list_normalized' in df_filtered.columns:
        df_filtered['ent_list'] = df_filtered['__ent_list_normalized']
    if 'user_id' not in df_filtered.columns:
        df_filtered = ensure_user_id_column(df_filtered)

    # Build User × Entitlement matrix only for filtered users
    try:
        uim = build_user_item_matrix(df_filtered)
    except KeyError as e:
        st.error(f"Pipeline expected column missing: {e}. Available columns: {list(df_filtered.columns)}")
        st.stop()

    st.write("User × Entitlement matrix shape (filtered):", uim.shape)
    st.dataframe(uim, use_container_width=True, height=300)

    # Compute min_role_size relative to filtered vocabulary
    total_unique_ents = len(uim.columns)
    min_role_size = max(1, math.ceil(total_unique_ents * ROLE_MIN_SHARE))
    st.sidebar.write(f"Role min size (entitlement count) set to: {min_role_size} (={ROLE_MIN_SHARE*100:.0f}% of filtered vocabulary)")

    # Save for inference later
    st.session_state['min_role_size'] = min_role_size
    st.session_state['total_unique_ents'] = total_unique_ents

    # Train/test split
    train_df, test_df = train_test_split_matrix(uim, test_size=TEST_SIZE)
    st.write(f"Train users: {len(train_df)}, Test users: {len(test_df)}")

    # Apriori on train set
    frequent = apriori_candidate_roles(train_df, min_support=MIN_SUPPORT)
    if frequent is None or frequent.empty:
        st.warning("No frequent itemsets found on filtered train data — try lowering min support.")
    else:
        fc = frequent.copy()
        fc["Entitlements"] = fc["itemsets"].apply(lambda x: ", ".join(sorted(list(x))))
        fc["Support"] = fc["support"].round(4)
        fc["#Entitlements"] = fc["length"]
        st.subheader("Frequent itemsets (top 20)")
        st.dataframe(fc[["Support", "Entitlements", "#Entitlements"]].head(20), use_container_width=True)

    # Candidate roles (apriori + clusters)
    apriori_roles = suggest_roles_from_apriori(frequent, train_df, min_popularity=MIN_POPULARITY, min_role_size=min_role_size)
    clusters_map = cluster_roles(train_df, n_clusters=KMEANS_N, method=CLUSTER_METHOD, eps=DBSCAN_EPS)
    cluster_roles_list = suggest_roles_from_clusters(clusters_map, train_df, min_popularity=MIN_POPULARITY, min_role_size=min_role_size)

    combined = apriori_roles + cluster_roles_list
    unique_candidates = dedupe_candidates(combined)

    # Enrich candidates
    enriched = []
    if 'EmpID' in df_filtered.columns:
        df_filtered['user_id'] = df_filtered['EmpID'].astype(str)
    elif 'user_id' in df_filtered.columns:
        df_filtered['user_id'] = df_filtered['user_id'].astype(str)
    else:
        df_filtered = df_filtered.reset_index().rename(columns={'index': 'user_id'}); df_filtered['user_id'] = df_filtered['user_id'].astype(str)
    job_map = {str(r['user_id']): (r.get('JobTitle') or '').strip() for _, r in df_filtered.iterrows()}
    dept_map = {str(r['user_id']): (r.get('Department') or '').strip() for _, r in df_filtered.iterrows()}

    for c in unique_candidates:
        entlist = c.get('entitlements', [])
        pop = evaluate_candidate_set(entlist, train_df)['popularity']
        cols = [e for e in entlist if e in train_df.columns]
        if cols:
            mask = (train_df[cols].sum(axis=1) == len(cols))
            members = train_df.index[mask].tolist()
        else:
            members = []
        jt_counter = Counter(job_map.get(str(uid), '') for uid in members)
        dept_counter = Counter(dept_map.get(str(uid), '') for uid in members)
        top_job = jt_counter.most_common(1)[0][0] if jt_counter else ''
        top_dept = dept_counter.most_common(1)[0][0] if dept_counter else ''
        role_name = f"{top_job} ({top_dept})" if top_job else f"InferredRole_{len(enriched)+1}"
        enriched.append({
            'entitlements': entlist,
            'popularity': pop,
            'members': members,
            'role_name': role_name,
            'ent_count': len(entlist),
            'source': c.get('source', 'apriori')
        })

    # Dedupe by role_name, keep highest popularity
    rolename_map = {}
    for e in enriched:
        rn = e.get('role_name') or ", ".join(e.get('entitlements', [])[:3])
        if rn not in rolename_map or e.get('popularity', 0) > rolename_map[rn].get('popularity', 0):
            rolename_map[rn] = e
    final_candidates = list(rolename_map.values())

    # Save outputs to session
    st.session_state['train_df'] = train_df
    st.session_state['test_df'] = test_df
    st.session_state['candidates'] = final_candidates
    st.session_state['role_members_map'], st.session_state['user_roles_map'] = assign_roles_to_users(final_candidates, train_df, assignment_fraction=ROLE_ASSIGN_FRAC)

    st.success(f"Discovered {len(final_candidates)} candidate roles after dedupe & filters (unique role names)")

    # Candidate summary
    if final_candidates:
        df_show = []
        for c in final_candidates:
            df_show.append({
                'Role Name': c['role_name'],
                '#Entitlements': c['ent_count'],
                'Popularity (users)': c['popularity'],
                'Source': c['source'],
                'Sample Members': ", ".join(map(str, c['members'][:5]))
            })
        st.subheader("Candidate roles (summary)")
        st.dataframe(pd.DataFrame(df_show).sort_values('Popularity (users)', ascending=False).head(200), use_container_width=True)

    # Assigned roles counts chart
    role_members_map, user_roles_map = st.session_state['role_members_map'], st.session_state['user_roles_map']
    chart_rows = []
    for key, members in role_members_map.items():
        role_name = next((c['role_name'] for c in final_candidates if tuple(c['entitlements']) == key), str(key)[:30])
        chart_rows.append({'Role': role_name, 'Members Count': len(members)})
    if chart_rows:
        df_chart = pd.DataFrame(chart_rows).sort_values('Members Count', ascending=False)
        st.subheader("Assigned roles - members count (unique role names)")
        st.dataframe(df_chart.head(50), use_container_width=True)
        st.bar_chart(df_chart.set_index('Role')['Members Count'].head(20))

    # Evaluation on test set
    eval_rows = []
    for c in final_candidates:
        entset = set(c.get('entitlements', []))
        ent_in_test = [e for e in entset if e in test_df.columns] if 'test_df' in locals() else []
        if not ent_in_test:
            matched = 0
        else:
            mask = (test_df[ent_in_test].sum(axis=1) == len(ent_in_test))
            matched = int(mask.sum())
        eval_rows.append({
            'Role': c.get('role_name', ''),
            '#Entitlements': len(entset),
            'Popularity (train)': int(c.get('popularity', 0)),
            'Matched (test users)': int(matched),
            'Entitlements (sample)': ", ".join(list(entset)[:8])
        })
    st.subheader("Evaluation on test set (how many test users would be matched)")
    if eval_rows:
        df_eval = pd.DataFrame(eval_rows).sort_values('Matched (test users)', ascending=False)
        st.dataframe(df_eval.head(100), use_container_width=True)
    else:
        st.info("No candidate roles to evaluate.")

    st.info("Pipeline complete. Use the discovered `candidates` and `train_df` in session_state for inference and visualization.")


# ---------------------------
# Inference (USE THE SAME FILTER TOKENS FROM PIPELINE)
# ---------------------------
st.sidebar.header('Inference (uses same tokens as pipeline filter)')

# These sliders are still inference-tunable (coverage/popularity/mode)
min_cov = st.sidebar.slider('Inference min coverage fraction', 0.1, 1.0, 0.5, 0.05)
min_pop_infer = st.sidebar.slider('Inference min popularity (final filter - min users)', 1, 200, 10, 1)
mode_infer = st.sidebar.selectbox("Matching mode", ["Strict subset (no extras allowed)", "Flexible (allow role extras)"])
max_extra_infer = 0
if mode_infer.startswith("Flexible"):
    max_extra_infer = st.sidebar.slider("Max extra entitlements allowed in suggested role (Flexible mode)", 0, 10, 1)

# Button to trigger inference (uses the same chosen_set that was used for filtering during pipeline run)
if st.sidebar.button('Suggest roles for filtered entitlement set'):

    # Ensure pipeline ran and produced candidates + train_df
    candidates = st.session_state.get('candidates')
    train_df = st.session_state.get('train_df')
    # The tokens actually used to filter during pipeline run (if the pipeline ran)
    chosen_set = st.session_state.get('chosen_filter_tokens')  # we will set this in pipeline code
    # Fallback: derive from the raw sidebar input if pipeline not run
    if chosen_set is None:
        # try to use the raw sidebar input used by pipeline (input_ents_raw)
        raw_items = [t for t in [x.strip() for x in input_ents_raw.split(",")] if t.strip()]
        chosen_set = set([str(x).strip().upper() for x in raw_items])

    # Guard
    if not chosen_set:
        st.error("No entitlement tokens available for inference. Run the pipeline with filter entitlements first or provide filter tokens in the sidebar.")
        st.stop()

    if candidates is None or train_df is None:
        st.error("Please run the pipeline first (click 'Run pipeline (filtered by provided entitlements)') to generate candidate roles.")
        st.stop()

    st.info(f"Using filter tokens for inference: {sorted(list(chosen_set))}")

    # Now perform matching same as before (Strict/Flexible)
    valid_suggestions = []
    reject_reasons = []
    min_role_size = st.session_state.get('min_role_size', 1)

    for c in candidates:
        role_ent_raw = [str(e).strip().upper() for e in c.get('entitlements', []) if str(e).strip()]
        if not role_ent_raw:
            reject_reasons.append((c.get('role_name', 'InferredRole'), 'empty_ent_set'))
            continue
        role_ent = set(role_ent_raw)

        # pipeline role size check
        if len(role_ent) < min_role_size:
            reject_reasons.append((c.get('role_name','InferredRole'), f'role_size_fail ({len(role_ent)} < {min_role_size})'))
            continue

        intersection = role_ent & chosen_set
        intersection_count = len(intersection)
        coverage_frac = intersection_count / max(1, len(chosen_set))
        extras_not_in_input = role_ent - chosen_set
        extras_count = len(extras_not_in_input)

        # Mode checks
        if mode_infer.startswith("Strict"):
            if not role_ent.issubset(chosen_set):
                reject_reasons.append((c.get('role_name','InferredRole'), f'strict_subset_fail (extras={extras_count})'))
                continue
        else:  # Flexible
            if chosen_set.issubset(role_ent):
                if extras_count > max_extra_infer:
                    reject_reasons.append((c.get('role_name','InferredRole'), f'too_many_extras ({extras_count} > {max_extra_infer})'))
                    continue
            else:
                if coverage_frac + 1e-9 < min_cov:
                    reject_reasons.append((c.get('role_name','InferredRole'), f'coverage_fail ({coverage_frac:.2f} < {min_cov:.2f})'))
                    continue

        # popularity check
        if c.get('popularity', 0) < min_pop_infer:
            reject_reasons.append((c.get('role_name','InferredRole'), f'popularity_fail ({c.get("popularity",0)} < {min_pop_infer})'))
            continue

        # accepted
        cand = dict(c)
        cand['role_ent_list'] = role_ent_raw
        cand['intersection_count'] = intersection_count
        cand['coverage_fraction'] = coverage_frac
        cand['extras'] = list(extras_not_in_input)
        cand['extras_count'] = extras_count
        valid_suggestions.append(cand)

    # dedupe by role_name (keep highest popularity)
    unique = {}
    for s in valid_suggestions:
        rn = s.get('role_name') or ", ".join(s.get('role_ent_list', [])[:3])
        if rn not in unique or s.get('popularity',0) > unique[rn].get('popularity',0):
            unique[rn] = s
    final_suggestions = list(unique.values())

    if not final_suggestions:
        st.warning("No roles match the constraints. Diagnostics below (why candidates were rejected).")
        if reject_reasons:
            df_rej = pd.DataFrame(reject_reasons, columns=['Role (example)','Reject reason']).head(200)
            st.dataframe(df_rej, use_container_width=True)
        st.info("Try lowering Coverage %, decreasing min popularity, switch to Flexible mode, or increase Max extras.")
        st.stop()

    # Sort and show
    final_suggestions = sorted(final_suggestions, key=lambda r: (-r['coverage_fraction'], -r.get('popularity',0)))
    display_rows = []
    for s in final_suggestions:
        display_rows.append({
            'Suggested Role': s.get('role_name'),
            'Ent Count': len(s.get('role_ent_list',[])),
            'Coverage %': f"{s['coverage_fraction']*100:.1f}%",
            'Match Count': s['intersection_count'],
            'Popularity (users)': int(s.get('popularity',0)),
            'Extras Count': s.get('extras_count',0),
            'Source': s.get('source','')
        })
    df_display = pd.DataFrame(display_rows)
    st.subheader(f"Suggested Roles — {len(final_suggestions)} match(es)")
    st.dataframe(df_display, use_container_width=True)

    # Per-role charts (same as before)
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    bar_color = "#0ea5e9"
    edge_color = "#0369a1"
    text_inside_color = "white"
    text_small_color = "#222"

    # role_members_map created during pipeline
    session_role_members = st.session_state.get('role_members_map', {})

    assignment_frac = globals().get('ROLE_ASSIGN_FRAC', None)
    if assignment_frac is None:
        assignment_frac = 1.0

    for s in valid_suggestions:
        role_name = s.get('role_name', 'InferredRole')
        role_ents = s.get('role_ent_list', [])
        popularity = int(s.get('popularity', 0)) or 0

        # 1) Prefer using role_members_map
        members = None
        key = tuple(role_ents)
        if key in session_role_members:
            members = session_role_members.get(key, [])
            baseline = len(members)
            raw_counts = []
            if baseline > 0:
                for ent in role_ents:
                    if ent in train_df.columns:
                        cnt = int(train_df.loc[members, ent].sum())
                    else:
                        cnt = 0
                    raw_counts.append(cnt)
            else:
                raw_counts = [int(train_df[ent].sum()) if ent in train_df.columns else 0 for ent in role_ents]
                baseline = max(1, max(raw_counts)) if raw_counts else 1
        else:
            # fallback: assignment-based membership
            if role_ents and all(e in train_df.columns for e in role_ents):
                min_required = max(1, int(math.ceil(assignment_frac * len(role_ents))))
                mask = (train_df[role_ents].sum(axis=1) >= min_required)
                members = train_df.index[mask].tolist()
                baseline = len(members)
                raw_counts = [int(train_df.loc[members, ent].sum()) if ent in train_df.columns else 0 for ent in role_ents]
                if baseline == 0:
                    raw_counts = [int(train_df[ent].sum()) if ent in train_df.columns else 0 for ent in role_ents]
                    baseline = max(1, max(raw_counts)) if raw_counts else 1
            else:
                raw_counts = [int(train_df[ent].sum()) if ent in train_df.columns else 0 for ent in role_ents]
                baseline = max(1, max(raw_counts)) if raw_counts else 1

        # percentages relative to baseline
        percents = [min((rc / baseline) * 100.0, 100.0) for rc in raw_counts]

        # --- 🔽 Sort entitlements by % descending
        ent_order = sorted(zip(role_ents, percents, raw_counts), key=lambda x: -x[1])
        role_ents_sorted, percents_sorted, raw_counts_sorted = zip(*ent_order)

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(10, 3.6), dpi=100)
        x = np.arange(len(role_ents_sorted))
        bars = ax.bar(x, percents_sorted, width=0.65,
                    edgecolor=edge_color, linewidth=1.4,
                    color=bar_color, zorder=3)

        ax.set_ylim(0, 110)
        ax.set_ylabel('Popularity %', fontsize=11)
        ax.set_title(f'{role_name} — entitlement prevalence (baseline = {baseline} members)',
                    fontsize=12, pad=8)

        ax.set_xticks(x)
        ax.set_xticklabels(role_ents_sorted, rotation=0, ha='center', fontsize=10)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # annotate bars
        for rect, pct, rc in zip(bars, percents_sorted, raw_counts_sorted):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, max(3, height*0.5), f"{pct:.0f}%",
                    ha='center', va='center', color=text_inside_color,
                    fontsize=12, fontweight='bold')
            ax.text(rect.get_x() + rect.get_width()/2, height + 3, f"{rc} members",
                    ha='center', va='bottom', color=text_small_color, fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)



