# # Sidebar controls
# DATA_PATH = st.sidebar.text_input('CSV file path', value=str(ROOT / "data" / "entitlements_ready.csv"))
# MIN_SUPPORT = st.sidebar.slider('Apriori min support', 0.001, 0.2, 0.01, 0.001)
# MIN_POPULARITY = st.sidebar.slider('Min role popularity (train filter)', 1, 200, 10, 1)
# MIN_COVERAGE = st.sidebar.slider('Inference: min coverage fraction (role must cover this fraction of requested entitlements)', 0.1, 1.0, 0.5, 0.05)
# CLUSTER_METHOD = st.sidebar.selectbox('Clustering method', ['kmeans', 'dbscan'])
# KMEANS_N = st.sidebar.slider('KMeans n_clusters', 2, 20, 5)
# DBSCAN_EPS = st.sidebar.slider('DBSCAN eps', 0.1, 1.0, 0.5, 0.05)
# TEST_SIZE = st.sidebar.slider('Test size fraction', 0.05, 0.5, 0.2, 0.05)

# # New: minimum role size defined as fraction of TOTAL DISTINCT entitlements in the sheet
# ROLE_MIN_SHARE = st.sidebar.slider('Role min share of total entitlements (fraction)', 0.0, 1.0, 0.5, 0.05)
# # assignment fraction — how much of role entitlements a user must have to be assigned that role
# ROLE_ASSIGN_FRAC = st.sidebar.slider('Assignment fraction (user must have >= this of role entitlements)', 0.1, 1.0, 1.0, 0.05)




# ui/role_mining_app.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd
import math
from collections import Counter

from role_mining.data_pipeline import load_csv, build_user_item_matrix, train_test_split_matrix
from role_mining.mining_models import (
    apriori_candidate_roles, cluster_roles,
    suggest_roles_from_apriori, suggest_roles_from_clusters,
    dedupe_candidates, match_roles_to_input, assign_roles_to_users, evaluate_candidate_set
)

st.set_page_config(page_title='Role Mining', layout='wide')
st.title('🔎 Role Mining — Apriori + Clustering (No LLM)')

# -----------------------------
# Sidebar (user-friendly)
# -----------------------------
st.sidebar.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #16a34a;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        border: 1px solid #0d6832;
    }
    div.stButton > button:first-child:hover {
        background-color: #15803d;
        border: 1px solid #0d6832;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.header("🔧 Role Mining Controls")

preset = st.sidebar.radio(
    "Quick presets (one-click tuning):",
    ("Default (balanced)", "Conservative (strict)", "Lenient (more suggestions)"),
    index=0
)

if preset == "Default (balanced)":
    preset_support = 0.01
    preset_popularity = 10
    preset_role_share = 0.5
    preset_assign_frac = 1.0
    preset_min_coverage = 0.5
elif preset == "Conservative (strict)":
    preset_support = 0.05
    preset_popularity = 20
    preset_role_share = 0.6
    preset_assign_frac = 1.0
    preset_min_coverage = 0.7
else:
    preset_support = 0.005
    preset_popularity = 3
    preset_role_share = 0.3
    preset_assign_frac = 0.8
    preset_min_coverage = 0.4

st.sidebar.markdown("**Data**")
DATA_PATH = st.sidebar.text_input(
    "CSV file path",
    value=str(ROOT / "data" / "entitlements_ready.csv"),
    help="Path to the CSV file that contains user entitlements. Must include an 'All_Entitlements' column."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Apriori (frequent patterns)**")

MIN_SUPPORT = st.sidebar.slider(
    "Min support — how common a pattern should be",
    min_value=0.001, max_value=0.5, value=float(preset_support), step=0.001,
    help="Fraction of users that must share an entitlement set for it to be considered (e.g. 0.01 = 1% of users). Lower → more matches; higher → fewer, stronger matches."
)
with st.sidebar.expander("Example / Tip: What is support?"):
    st.write("- If support = 0.01 and you have 1,000 users → the itemset must appear in ≥ 10 users.")
    st.write("- Use smaller values for large datasets or many distinct entitlements.")

MIN_POPULARITY = st.sidebar.slider(
    "Min role popularity — minimum users for a role",
    1, 500, int(preset_popularity), step=1,
    help="A discovered role must apply to at least this many users to be considered valid (absolute count)."
)
with st.sidebar.expander("Why popularity matters"):
    st.write("- Prevents tiny, one-off 'roles' created by noise.")
    st.write("- If your org is small, lower this value (e.g. 5).")

st.sidebar.markdown("---")
st.sidebar.markdown("**Role size & assignment**")

ROLE_MIN_SHARE = st.sidebar.slider(
    "Role size (fraction of all entitlements)",
    0.0, 1.0, float(preset_role_share), step=0.05,
    help="Controls how large a role should be relative to the total unique entitlements in the dataset. Example: 0.5 means a role must have at least 50% of the distinct entitlements in the sheet."
)
with st.sidebar.expander("Example: role size rule"):
    st.write("- If your dataset has 20 unique entitlements and slider = 0.5 → a role must contain at least 10 entitlements.")
    st.write("- Lower this number to allow smaller, more granular roles.")

ROLE_ASSIGN_FRAC = st.sidebar.slider(
    "Assignment threshold — how much of a role a user must have",
    0.1, 1.0, float(preset_assign_frac), step=0.05,
    help="When assigning discovered roles to users, the user must have ≥ this fraction of the role's entitlements to be considered a member (1.0 = must have all entitlements)."
)
with st.sidebar.expander("Why assignment threshold helps"):
    st.write("- 1.0 => strict assignment (user must have the full role entitlement set).")
    st.write("- Lowering to 0.8 allows partial matches (helpful if real users miss 1–2 entitlements).")

st.sidebar.markdown("---")
st.sidebar.markdown("**Discovery & evaluation**")

CLUSTER_METHOD = st.sidebar.selectbox(
    "Clustering method (alternate discovery)",
    ['kmeans', 'dbscan'],
    help="Use clustering to find groups of similar users as alternative candidate roles. KMeans groups into K clusters; DBSCAN finds dense clusters."
)

KMEANS_N = st.sidebar.slider(
    "KMeans: number of clusters",
    2, 40, 5, step=1,
    help="How many clusters to create when KMeans is selected. More clusters → more, smaller role-like groups."
)

DBSCAN_EPS = st.sidebar.slider(
    "DBSCAN: eps (distance threshold)",
    0.05, 1.0, 0.5, step=0.01,
    help="How close users must be to be in same DBSCAN cluster. Lower → tighter clusters; higher → looser clusters."
)

TEST_SIZE = st.sidebar.slider(
    "Test size (hold-out fraction)",
    0.05, 0.5, 0.2, step=0.05,
    help="Fraction of users held out for testing/evaluation (pipeline uses train to discover roles and test to evaluate how well roles generalize)."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Inference (when a user requests access)**")

MIN_COVERAGE = st.sidebar.slider(
    "Inference coverage — how much of the requested entitlements must a role cover",
    0.1, 1.0, float(preset_min_coverage), step=0.05,
    help="When a user provides a set of requested entitlements, a suggested role must contain at least this fraction of those entitlements to be considered a match."
)
with st.sidebar.expander("Example: inference coverage"):
    st.write("- If a user asks for 4 entitlements and coverage = 0.5, a role must contain at least 2 of those to be suggested.")

MIN_POP_INFER = st.sidebar.slider(
    "Inference min popularity",
    1, 500, int(preset_popularity), step=1,
    help="Minimum number of users (popularity) a suggested role must have in the training set to be suggested to admins."
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Start with the **Default (balanced)** preset. If you see no roles, switch to **Lenient** or reduce 'Role size' and 'Min role popularity'.")

ENT_COL_HINT = "All_Entitlements"  # your CSV column

# run pipeline
if st.button('Run pipeline'):
    # load raw CSV (for mapping job/department)
    try:
        df_raw = pd.read_csv(DATA_PATH, dtype=str)
    except Exception as e:
        st.error(f"Could not open CSV at {DATA_PATH}: {e}")
        st.stop()

    # parse entitlements using loader
    try:
        df_parsed = load_csv(DATA_PATH, ent_col_hint=ENT_COL_HINT)
    except Exception as e:
        st.error(f'Failed to parse CSV: {e}')
        st.stop()

    st.success(f'Loaded {len(df_parsed)} identities (parsed entitlements)')

    # Build user-item matrix
    uim = build_user_item_matrix(df_parsed)
    st.write('User × Entitlement matrix shape:', uim.shape)
    # show full matrix (warning: can be large)
    st.dataframe(uim, width='stretch')

    # Total distinct entitlements in the sheet
    total_unique_ents = len(uim.columns)
    st.sidebar.write(f"Total distinct entitlements (vocabulary): {total_unique_ents}")

    # compute min_role_size as fraction of total_unique_ents (user requested behavior)
    min_role_size = max(1, math.ceil(total_unique_ents * ROLE_MIN_SHARE))
    st.sidebar.write(f"Role min size (entitlement count) set to: {min_role_size} (={ROLE_MIN_SHARE*100:.0f}% of total entitlements)")

    # Train-test split
    train_df, test_df = train_test_split_matrix(uim, test_size=TEST_SIZE)
    st.write(f'Train users: {len(train_df)}, Test users: {len(test_df)}')

    # Apriori frequent itemsets
    frequent = apriori_candidate_roles(train_df, min_support=MIN_SUPPORT)
    st.write('Frequent itemsets (top 20)')
    if frequent is None or frequent.empty:
        st.warning("No frequent itemsets found — try lowering min support.")
    else:
        # Convert frozenset to readable string
        frequent_clean = frequent.copy()
        frequent_clean["Entitlements"] = frequent_clean["itemsets"].apply(lambda x: ", ".join(sorted(list(x))))
        frequent_clean["Support"] = frequent_clean["support"].round(4)
        frequent_clean["#Entitlements"] = frequent_clean["length"]
        display_freq = frequent_clean[["Support", "Entitlements", "#Entitlements"]]
        st.dataframe(display_freq.head(20), width='stretch')

    # Generate candidate roles (apriori + clusters) using min_role_size (absolute ent count)
    apriori_roles = suggest_roles_from_apriori(frequent, train_df, min_popularity=MIN_POPULARITY, min_role_size=min_role_size)
    clusters_map = cluster_roles(train_df, n_clusters=KMEANS_N, method=CLUSTER_METHOD, eps=DBSCAN_EPS)
    cluster_roles_list = suggest_roles_from_clusters(clusters_map, train_df, min_popularity=MIN_POPULARITY, min_role_size=min_role_size)

    # Combine + dedupe by entitlement set
    combined = apriori_roles + cluster_roles_list
    unique_candidates = dedupe_candidates(combined)

    # compute final popularity on train set and add members and a role_name
    enriched = []
    # create job/department maps from raw data for role naming
    if 'EmpID' in df_raw.columns:
        df_raw['user_id'] = df_raw['EmpID'].astype(str)
    elif 'user_id' in df_raw.columns:
        df_raw['user_id'] = df_raw['user_id'].astype(str)
    else:
        df_raw = df_raw.reset_index().rename(columns={'index': 'user_id'}); df_raw['user_id'] = df_raw['user_id'].astype(str)
    job_map = {str(r['user_id']): (r.get('JobTitle') or '').strip() for _, r in df_raw.iterrows()}
    dept_map = {str(r['user_id']): (r.get('Department') or '').strip() for _, r in df_raw.iterrows()}

    for c in unique_candidates:
        entlist = c.get('entitlements', [])
        pop = evaluate_candidate_set(entlist, train_df)['popularity']
        # determine members (strict all entitlements)
        cols = [e for e in entlist if e in train_df.columns]
        if cols:
            mask = (train_df[cols].sum(axis=1) == len(cols))
            members = train_df.index[mask].tolist()
        else:
            members = []
        # role naming from top jobtitle among members
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

    # --- NEW: dedupe by role_name to avoid duplicate displayed roles ---
    rolename_map = {}
    for e in enriched:
        rn = e.get('role_name') or ", ".join(e.get('entitlements', [])[:3
])
        if rn not in rolename_map:
            rolename_map[rn] = e
        else:
            if e.get('popularity', 0) > rolename_map[rn].get('popularity', 0):
                rolename_map[rn] = e
    final_candidates = list(rolename_map.values())

    # store both train_df and candidates in session for later inference
    st.session_state['train_df'] = train_df
    st.session_state['candidates'] = final_candidates

    st.success(f"Discovered {len(final_candidates)} candidate roles after dedupe & filters (unique role names)")


    # Show candidate roles summary (friendly columns)
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
        st.dataframe(pd.DataFrame(df_show).sort_values('Popularity (users)', ascending=False).head(200), width='stretch')

    # Assign roles to train-set users based on ROLE_ASSIGN_FRAC
    role_members_map, user_roles_map = assign_roles_to_users(final_candidates, train_df, assignment_fraction=ROLE_ASSIGN_FRAC)
    # Build counts per role (role key -> count)
    role_counts = {tuple(k): len(v) for k, v in role_members_map.items()}
    # Prepare display series for chart (use role_name mapping)
    role_name_map = {tuple(c['entitlements']): c['role_name'] for c in final_candidates}
    chart_rows = []
    for key, count in role_counts.items():
        chart_rows.append({'Role Key': key, 'Role': role_name_map.get(tuple(key), str(key)[:30]), 'Members Count': count})
    if chart_rows:
        df_chart = pd.DataFrame(chart_rows).sort_values('Members Count', ascending=False)
        st.subheader("Assigned roles - members count (unique role names)")
        st.dataframe(df_chart[['Role', 'Members Count']].head(50), width='stretch')
        # bar chart (top 20)
        # st.bar_chart(df_chart.set_index('Role')['Members Count'].head(20))

    # Save role assignments in session for later use if desired
    st.session_state['role_members_map'] = role_members_map
    st.session_state['user_roles_map'] = user_roles_map

        # Evaluate on test set - friendly table (safe: always include keys)
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
    if not eval_rows:
        st.info("No candidate roles to evaluate.")
    else:
        df_eval = pd.DataFrame(eval_rows)
        # Defensive sort: use 'Matched (test users)' if present, else fall back to popularity
        if 'Matched (test users)' in df_eval.columns:
            df_eval = df_eval.sort_values('Matched (test users)', ascending=False)
        else:
            df_eval = df_eval.sort_values('Popularity (train)', ascending=False)
        st.dataframe(df_eval.head(100), width='stretch')


# ---------------------------
# Inference: suggest roles for given entitlements
# ---------------------------
# st.sidebar.header('Inference')
# input_ents = st.sidebar.text_area('Enter entitlements (comma-separated)', value='ENT_READ_FINDB, ENT_VIEW_REPORTS')
# min_cov = st.sidebar.slider('Inference min coverage fraction', 0.1, 1.0, 0.5, 0.05)
# min_pop_infer = st.sidebar.slider('Inference min popularity (final filter)', 1, 200, 10, 1)

# if st.sidebar.button('Suggest roles for this entitlement set'):
#     candidates = st.session_state.get('candidates', None)
#     ent_list = [e.strip() for e in input_ents.split(',') if e.strip()]
#     if candidates is None:
#         st.error("Run pipeline first to generate candidate roles.")
#     else:
#         st.info(f"Total candidate roles available: {len(candidates)}")
#         matches = match_roles_to_input(ent_list, candidates, min_match_frac=min_cov)
#         matches = [m for m in matches if m.get('popularity', 0) >= min_pop_infer]

#         # de-duplicate by role_name (keep highest popularity)
#         seen_role_names = {}
#         for m in matches:
#             rn = m.get('role_name') if m.get('role_name') else ", ".join(m.get('entitlements', [])[:3])
#             if rn not in seen_role_names:
#                 seen_role_names[rn] = m
#             else:
#                 if m.get('popularity', 0) > seen_role_names[rn].get('popularity', 0):
#                     seen_role_names[rn] = m
#         unique_matches = list(seen_role_names.values())

#         if not unique_matches:
#             st.warning("No candidate role matches constraints. Try lowering min_support/min_popularity or adjust inference sliders.")
#             st.info("Tip: set 'Role min share' lower or reduce 'Min role popularity' if dataset is sparse.")
#         else:
#             st.success(f"Found {len(unique_matches)} matching role(s) (unique role names)")

#             # --- Helper / legend for table columns ---
#             with st.expander("ℹ️ What these columns mean (click to expand)"):
#                 st.markdown(
#                     """
#                     **Suggested Role** — Human-friendly role name inferred from job titles of users who matched the entitlement set.  
#                     **Entitlements (sample)** — A short sample list of entitlements in the role (first 8). Click *Top suggestion* for the full list.  
#                     **Entitlements Count** — Number of entitlements that define this role.  
#                     **Match Count** — How many of the entitlements you provided match this role.  
#                     **Coverage %** — Fraction of your requested entitlements that this role contains (higher is better).  
#                     **Popularity** — How many users in the training set have the full role entitlement set. This indicates how common the role is.  
#                     **Source** — How the role was discovered (e.g. `apriori`, `cluster_3`).
#                     """
#                 )

#             # Build display-friendly DataFrame
#             display_rows = []
#             for m in unique_matches:
#                 ent_sample = m.get('entitlements', [])[:8]
#                 display_rows.append({
#                     'Suggested Role': m.get('role_name'),
#                     'Entitlements (sample)': "\n".join(ent_sample) if ent_sample else "",
#                     'Entitlements Count': len(m.get('entitlements', [])),
#                     'Match Count': m.get('match_count', 0),
#                     'Coverage %': f"{m.get('match_fraction', 0)*100:.1f}%",
#                     'Popularity': int(m.get('popularity', 0)),
#                     'Source': m.get('source', '')
#                 })

#             df_display = pd.DataFrame(display_rows)

#             # Conditional styling for Coverage %
#             def coverage_style(cell):
#                 try:
#                     val = float(cell.strip('%'))
#                 except Exception:
#                     return ''
#                 if val >= 80:
#                     return 'background-color: #16a34a; color: white;'   # green
#                 if val >= 50:
#                     return 'background-color: #f59e0b; color: black;'   # yellow
#                 return 'background-color: #dc2626; color: white;'       # red

#             st.subheader("📋 Suggested Roles")
#             styled = df_display.style.applymap(coverage_style, subset=['Coverage %'])
#             # show as scrollable and wider
#             st.dataframe(styled, use_container_width=True, height=380)

#             # Download button for admins to export suggestions as CSV
#             csv = df_display.to_csv(index=False).encode('utf-8')
#             st.download_button(label="⬇️ Download suggestions (CSV)", data=csv, file_name="suggested_roles.csv", mime="text/csv")

#             # Detailed top suggestion (full entitlements)
#             top = sorted(unique_matches, key=lambda r: (-r.get('match_fraction',0), -r.get('popularity',0)))[0]
#             st.subheader(f"⭐ Top suggestion: {top.get('role_name')}")
#             st.markdown("**Full entitlements for this role:**")
#             for ent in top.get('entitlements', []):
#                 st.markdown(f"- `{ent}`")
#             st.markdown(f"**Coverage:** {top.get('match_fraction')*100:.1f}%")
#             st.markdown(f"**Popularity:** {top.get('popularity')} users (in training set)")

# ---------------------------
# Inference (diagnostics + strict/flexible mode)
# ---------------------------
st.sidebar.header('Inference')
input_ents = st.sidebar.text_area('Enter entitlements (comma-separated)', value='ENT_READ_FINDB, ENT_VIEW_REPORTS')
min_cov = st.sidebar.slider('Inference min coverage fraction (role must cover ≥ this of provided entitlements)', 0.1, 1.0, 0.5, 0.05)
min_pop_infer = st.sidebar.slider('Inference min popularity (final filter - min users)', 1, 500, 10, 1)

# New: mode toggle + extras allowance for flexible mode
mode = st.sidebar.selectbox("Matching mode", ["Strict subset (no extras allowed)", "Flexible (allow role extras)"])
max_extra = 0
if mode.startswith("Flexible"):
    max_extra = st.sidebar.slider("Max extra entitlements allowed in suggested role", 0, 10, 1, 1,
                                  help="Role may contain up to this many entitlements that were NOT provided by the user (use 0 to emulate strict).")

if st.sidebar.button('Suggest roles for this entitlement set'):
    candidates = st.session_state.get('candidates')
    train_df = st.session_state.get('train_df')
    if candidates is None or train_df is None:
        st.error("Please run the pipeline first (click 'Run pipeline') to generate candidate roles and training data.")
        st.stop()

    # Normalize input tokens (trim + uppercase)
    input_list_raw = [e for e in [x.strip() for x in input_ents.split(',')] if e]
    if not input_list_raw:
        st.error("Please enter at least one entitlement in the Inference box.")
        st.stop()
    # Keep normalization conservative but visible
    input_list = [str(e).strip() for e in input_list_raw]
    input_set = set(input_list)

    st.info(f"Searching roles (mode: {mode}) that match the provided entitlements ({len(input_set)} provided).")
    st.caption("Tip: tokens must match exactly. Leading/trailing spaces or different prefixes/casing will prevent matches.")

    # # Diagnostics accumulators
    # filtered_reasons = []  # list of tuples (role_name, reason)
    # valid_suggestions = []

    # for c in candidates:
    #     role_ent_raw = [str(e).strip() for e in c.get('entitlements', []) if str(e).strip()]
    #     if not role_ent_raw:
    #         filtered_reasons.append((c.get('role_name', 'InferredRole'), "empty ent set"))
    #         continue
    #     role_ent = set(role_ent_raw)

    #     # check token overlap / normalization issues: show difference sets
    #     extras_not_in_input = role_ent - input_set
    #     intersection_count = len(role_ent & input_set)

    #     # Mode checks
    #     if mode.startswith("Strict"):
    #         # strict subset: role_ent must be subset of input_set
    #         if not role_ent.issubset(input_set):
    #             filtered_reasons.append((c.get('role_name', 'InferredRole'), f"subset_fail (extras: {len(extras_not_in_input)})"))
    #             continue
    #     else:
    #         # Flexible: allow up to max_extra entitlements that are outside input_set
    #         if len(extras_not_in_input) > max_extra:
    #             filtered_reasons.append((c.get('role_name', 'InferredRole'), f"too_many_extras ({len(extras_not_in_input)} > {max_extra})"))
    #             continue

    #     # coverage: fraction of provided entitlements that role covers
    #     coverage_frac = len(role_ent & input_set) / max(1, len(input_set))
    #     if coverage_frac + 1e-9 < min_cov:
    #         filtered_reasons.append((c.get('role_name', 'InferredRole'), f"coverage_fail ({coverage_frac:.2f} < {min_cov})"))
    #         continue

    #     # popularity
    #     if c.get('popularity', 0) < min_pop_infer:
    #         filtered_reasons.append((c.get('role_name', 'InferredRole'), f"popularity_fail ({c.get('popularity',0)} < {min_pop_infer})"))
    #         continue

    #     # Passed all checks
    #     cand = dict(c)
    #     cand['coverage_fraction'] = coverage_frac
    #     cand['role_ent_list'] = role_ent_raw
    #     valid_suggestions.append(cand)

    # Diagnostics accumulators (ensure these are defined before this block)
    reject_reasons = []
    valid_suggestions = []
    min_role_size = st.session_state.get('min_role_size', 1)  # already stored by pipeline

    # For debug: how many user tokens are in vocab
    vocab = set(train_df.columns.tolist())
    missing_tokens = [t for t in input_list if t not in vocab]
    if missing_tokens:
        st.warning(f"These entitlements are not in vocabulary and won't match: {missing_tokens}")

    for c in candidates:
        # parse role entitlements from candidate
        role_ent_raw = [str(e).strip() for e in c.get('entitlements', []) if str(e).strip()]
        if not role_ent_raw:
            reject_reasons.append((c.get('role_name','InferredRole'), "empty ent set"))
            continue

        role_ent = set(role_ent_raw)

        # enforce pipeline role size
        if len(role_ent) < min_role_size:
            reject_reasons.append((c.get('role_name','InferredRole'), f"role_size_fail ({len(role_ent)} < {min_role_size})"))
            continue

        # compute intersection and extras
        intersection = role_ent & input_set
        intersection_count = len(intersection)
        extras = role_ent - input_set
        extras_count = len(extras)

        # compute coverage: fraction of user-provided entitlements that the role contains
        coverage_frac = intersection_count / max(1, len(input_set))

        # Decide acceptance based on mode
        if mode.startswith("Strict"):
            # Strict: role must be subset of the provided set (no extras allowed)
            if not role_ent.issubset(input_set):
                reject_reasons.append((c.get('role_name','InferredRole'), f"subset_fail (extras={extras_count})"))
                continue
            # also require coverage >= min_cov (usually will be 1.0 when subset)
            if coverage_frac + 1e-9 < min_cov:
                reject_reasons.append((c.get('role_name','InferredRole'), f"coverage_fail ({coverage_frac:.2f} < {min_cov:.2f})"))
                continue

        elif mode.startswith("Flexible"):
            # Flexible semantics (as you requested):
            # - role may contain entitlements not provided by the user, up to `max_extra`
            # - role must still cover at least `min_cov` fraction of the provided entitlements
            # Accept candidate only if both conditions hold.
            if coverage_frac + 1e-9 < min_cov:
                reject_reasons.append((c.get('role_name','InferredRole'), f"coverage_fail ({coverage_frac:.2f} < {min_cov:.2f})"))
                continue
            if extras_count > max_extra:
                reject_reasons.append((c.get('role_name','InferredRole'), f"too_many_extras ({extras_count} > {max_extra})"))
                continue

        else:
            # fallback: treat like relaxed coverage-based matching
            if coverage_frac + 1e-9 < min_cov:
                reject_reasons.append((c.get('role_name','InferredRole'), f"coverage_fail ({coverage_frac:.2f} < {min_cov:.2f})"))
                continue

        # final popularity filter
        if c.get('popularity', 0) < min_pop_infer:
            reject_reasons.append((c.get('role_name','InferredRole'), f"popularity_fail ({c.get('popularity',0)} < {min_pop_infer})"))
            continue

        # Candidate accepted — attach debug info
        cand = dict(c)
        cand['coverage_fraction'] = coverage_frac
        cand['intersection_count'] = intersection_count
        cand['extras_count'] = extras_count
        cand['extras'] = list(extras)
        cand['role_ent_list'] = role_ent_raw
        valid_suggestions.append(cand)


    # If none valid, show diagnostics to help user
    if not valid_suggestions:
        st.warning("No roles match the constraints.")
        st.markdown("**Diagnostics (why each candidate was rejected):**")
        # Show first 100 candidate reasons in a compact table
        reason_df = pd.DataFrame(reject_reasons, columns=["Role (example)", "Reject reason"]).head(200)
        if reason_df.empty:
            st.info("No candidates were evaluated - something may be wrong with the pipeline output.")
        else:
            st.dataframe(reason_df, use_container_width=True)
        st.info("Suggestions: lower Coverage %, reduce Min popularity, switch to Flexible mode or increase 'Max extra entitlements'.")
        st.stop()

    # Sort and show results
    valid_suggestions = sorted(valid_suggestions, key=lambda r: (-r['coverage_fraction'], -r.get('popularity', 0)))
    rows = []
    for s in valid_suggestions:
        rows.append({
            "Suggested Role": s.get('role_name', 'InferredRole'),
            "Role Entitlements": "\n".join(s.get('role_ent_list', [])[:12]),
            "Role Ent Count": len(s.get('role_ent_list', [])),
            "Coverage %": f"{s['coverage_fraction']*100:.1f}%",
            "Popularity": int(s.get('popularity', 0)),
            "Source": s.get('source', '')
        })
    df_suggest = pd.DataFrame(rows)

    st.subheader(f"Suggested Roles — {len(valid_suggestions)} match(es)")
    with st.expander("ℹ️ How suggestions are calculated (click to expand)"):
        st.markdown(
            "- **Strict subset**: suggested role must contain only entitlements from the set you provided (no extras).\n"
            "- **Flexible**: role may include up to *Max extra* entitlements that were NOT provided (use slider).\n"
            "- All suggested roles must cover ≥ Coverage % of your provided entitlements and meet min popularity.\n"
            "- If no results appear, the diagnostics table below shows why each candidate was rejected."
        )
    st.dataframe(df_suggest, use_container_width=True, height=220)

    # Small download
    # st.download_button("⬇️ Download suggestions", df_suggest.to_csv(index=False).encode('utf-8'), "suggestions.csv", "text/csv")

        # --- per-role charts (percent + raw members) ---
    import matplotlib.pyplot as plt
    import numpy as np

    bar_color = "#0ea5e9"
    edge_color = "#0369a1"
    text_inside_color = "white"
    text_small_color = "#222"

    # try to get role->members mapping from session (created when pipeline ran)
    session_role_members = st.session_state.get('role_members_map', {})

    # assignment fraction used previously for assigning roles (fallback default)
    assignment_frac = globals().get('ROLE_ASSIGN_FRAC', None)
    if assignment_frac is None:
        assignment_frac = 1.0

    for s in valid_suggestions:
        role_name = s.get('role_name', 'InferredRole')
        role_ents = s.get('role_ent_list', [])
        popularity = int(s.get('popularity', 0)) or 0

        # 1) Prefer using the explicit role_members_map computed during pipeline
        members = None
        key = tuple(role_ents)
        if key in session_role_members:
            members = session_role_members.get(key, [])
            baseline = len(members)
            # counts among the role members
            raw_counts = []
            if baseline > 0:
                for ent in role_ents:
                    if ent in train_df.columns:
                        # count among role members only
                        cnt = int(train_df.loc[members, ent].sum()) if len(members) > 0 else 0
                    else:
                        cnt = 0
                    raw_counts.append(cnt)
            else:
                # fallback to counts across train set if role_members_map empty
                for ent in role_ents:
                    cnt = int(train_df[ent].sum()) if ent in train_df.columns else 0
                    raw_counts.append(cnt)
                baseline = max(1, max(raw_counts)) if raw_counts else 1
        else:
            # 2) Fallback: try to derive members by assignment fraction over train_df
            # require at least ROLE_ASSIGN_FRAC of role entitlements to consider a user a member
            # (this handles flexible roles if session map isn't available)
            if role_ents and all(e in train_df.columns for e in role_ents):
                # users that have at least ceil(assignment_frac * len(role_ents)) entitlements of the role
                min_required = max(1, int(math.ceil(assignment_frac * len(role_ents))))
                mask = (train_df[role_ents].sum(axis=1) >= min_required)
                members = train_df.index[mask].tolist()
                baseline = len(members)
                raw_counts = [int(train_df.loc[members, ent].sum()) if ent in train_df.columns else 0 for ent in role_ents]
                if baseline == 0:
                    # fallback: use total counts across all users and baseline = max(counts)
                    raw_counts = [int(train_df[ent].sum()) if ent in train_df.columns else 0 for ent in role_ents]
                    baseline = max(1, max(raw_counts)) if raw_counts else 1
            else:
                # safe fallback: counts across entire train set, baseline = max(counts)
                raw_counts = [int(train_df[ent].sum()) if ent in train_df.columns else 0 for ent in role_ents]
                baseline = max(1, max(raw_counts)) if raw_counts else 1

        # compute percentages relative to baseline
        percents = [min((rc / baseline) * 100.0, 100.0) for rc in raw_counts]

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(10, 3.6), dpi=100)
        x = np.arange(len(role_ents))
        bars = ax.bar(x, percents, width=0.65, edgecolor=edge_color,
                      linewidth=1.4, color=bar_color, zorder=3)

        ax.set_ylim(0, 110)
        ax.set_ylabel('Popularity %', fontsize=11)
        ax.set_title(f'{role_name} — entitlement prevalence (baseline = {baseline} members)',
                     fontsize=12, pad=8)

        ax.set_xticks(x)
        # ✅ Keep entitlement labels horizontal and centered under bars
        ax.set_xticklabels(role_ents, rotation=0, ha='center', fontsize=10)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


        # Annotate bars: percent inside + raw count above
        for rect, pct, rc in zip(bars, percents, raw_counts):
            height = rect.get_height()
            # percent inside bar (use min y to avoid overlapping with very small bars)
            ax.text(rect.get_x() + rect.get_width()/2, max(3, height*0.5), f"{pct:.0f}%", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
            # raw count above bar
            ax.text(rect.get_x() + rect.get_width()/2, height + 3, f"{rc} members", ha='center', va='bottom', color=text_small_color, fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)







