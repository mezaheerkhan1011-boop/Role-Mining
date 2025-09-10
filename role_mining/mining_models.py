# role_mining/mining_models.py
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from collections import Counter
import math

def apriori_candidate_roles(user_item_df, min_support=0.05, min_len=1, max_len=None):
    df_bool = user_item_df.astype(bool)
    if df_bool.shape[0] == 0:
        return pd.DataFrame()
    frequent = apriori(df_bool, min_support=min_support, use_colnames=True)
    if frequent is None or frequent.empty:
        return frequent
    frequent['length'] = frequent['itemsets'].apply(lambda s: len(s))
    if max_len is not None:
        frequent = frequent[(frequent['length'] >= min_len) & (frequent['length'] <= max_len)]
    else:
        frequent = frequent[frequent['length'] >= min_len]
    freq_sorted = frequent.sort_values('support', ascending=False).reset_index(drop=True)
    return freq_sorted

def cluster_roles(user_item_df, n_clusters=5, method='kmeans', random_state=42, eps=0.5, min_samples=3):
    X = user_item_df.values.astype(float)
    if X.shape[0] == 0:
        return {}
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X)
    clusters = {}
    if method == 'kmeans':
        km = KMeans(n_clusters=max(1, n_clusters), random_state=random_state)
        labels = km.fit_predict(Xs)
        centers = km.cluster_centers_
        for cid in np.unique(labels):
            center = centers[cid]
            selected = list(user_item_df.columns[(center > 0.3)])
            clusters[int(cid)] = {'members': user_item_df.index[labels == cid].tolist(),
                                  'entitlements': selected,
                                  'size': int((labels == cid).sum())}
    elif method == 'dbscan':
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = db.fit_predict(Xs)
        for cid in np.unique(labels):
            if cid == -1:
                continue
            members = user_item_df.index[labels == cid].tolist()
            mean_vec = X[labels == cid].mean(axis=0)
            selected = list(user_item_df.columns[mean_vec > 0.3])
            clusters[int(cid)] = {'members': members, 'entitlements': selected, 'size': len(members)}
    else:
        raise ValueError('Unknown clustering method')
    return clusters

def evaluate_candidate_set(candidate_itemset, user_item_df):
    """
    candidate_itemset: iterable of entitlement names
    Returns popularity = number of users in user_item_df that have ALL entitlements in candidate_itemset
    """
    s = set(candidate_itemset)
    if len(s) == 0:
        return {'popularity': 0}
    cols = [c for c in s if c in user_item_df.columns]
    if not cols:
        return {'popularity': 0}
    mask = (user_item_df[cols].sum(axis=1) == len(cols))
    popularity = int(mask.sum())
    return {'popularity': popularity}

def suggest_roles_from_apriori(frequent_itemsets_df, user_item_df, min_popularity=10, min_role_size=0):
    """
    Convert frequent itemsets into candidate roles and filter by min_popularity.
    Additionally apply a min_role_size (absolute number of entitlements) filter.
    Returns list of dicts: {entitlements:list, support:float, popularity:int, source:'apriori'}
    """
    roles = []
    if frequent_itemsets_df is None or frequent_itemsets_df.empty:
        return roles
    for _, row in frequent_itemsets_df.iterrows():
        itemset = set(row['itemsets'])
        # apply min_role_size filter (number of entitlements)
        if len(itemset) < min_role_size:
            continue
        pop = evaluate_candidate_set(itemset, user_item_df)['popularity']
        if pop >= min_popularity:
            roles.append({'entitlements': sorted(itemset), 'support': float(row['support']), 'popularity': pop, 'source': 'apriori'})
    roles_sorted = sorted(roles, key=lambda r: (-r['popularity'], -r['support']))
    return roles_sorted

def suggest_roles_from_clusters(cluster_map, user_item_df, min_popularity=10, min_role_size=0):
    roles = []
    for cid, info in cluster_map.items():
        entset = set(info.get('entitlements', []))
        if len(entset) < min_role_size:
            continue
        pop = evaluate_candidate_set(entset, user_item_df)['popularity']
        if pop >= min_popularity:
            roles.append({'entitlements': sorted(entset), 'popularity': pop, 'source': f'cluster_{cid}'})
    roles_sorted = sorted(roles, key=lambda r: (-r['popularity'], -len(r['entitlements'])))
    return roles_sorted

def dedupe_candidates(candidate_list):
    """
    Deduplicate by entitlement set (order-insensitive).
    Returns list of unique candidate dicts.
    """
    seen = {}
    for c in candidate_list:
        key = tuple(sorted(c.get('entitlements', [])))
        if not key:
            continue
        if key not in seen:
            seen[key] = dict(c)
        else:
            # keep higher popularity/support
            seen[key]['popularity'] = max(seen[key].get('popularity', 0), c.get('popularity', 0))
            if c.get('support', 0) > seen[key].get('support', 0):
                seen[key]['support'] = c.get('support', seen[key].get('support'))
    return list(seen.values())

def match_roles_to_input(input_entitlements, candidate_roles, min_match_frac=0.5):
    """
    Returns roles that cover >= min_match_frac of the input_entitlements.
    Matching is case-insensitive and whitespace-trimmed.
    """
    def norm(s):
        return s.strip().lower()
    inp = set(norm(i) for i in input_entitlements if i and str(i).strip())
    if not inp:
        return []
    matches = []
    for role in candidate_roles:
        role_set = set(norm(e) for e in role.get('entitlements', []))
        match_count = len(inp & role_set)
        match_frac = match_count / len(inp)
        if match_frac >= min_match_frac and match_count > 0:
            entry = dict(role)
            entry['match_count'] = match_count
            entry['match_fraction'] = round(match_frac, 3)
            matches.append(entry)
    matches_sorted = sorted(matches, key=lambda r: (-r['match_fraction'], -r.get('popularity', 0)))
    return matches_sorted

def assign_roles_to_users(candidate_roles, user_item_df, assignment_fraction=1.0):
    """
    For each user in user_item_df, assign roles for which the user has >= assignment_fraction
    fraction of the role's entitlements. Return mapping role_key -> list(user_ids) and user->roles map.

    assignment_fraction: fraction of role entitlements the user must possess to be considered member.
                         (1.0 = user must have all entitlements of role)
    """
    role_members = {}
    user_roles = {uid: [] for uid in user_item_df.index}
    for c in candidate_roles:
        ents = c.get('entitlements', [])
        if not ents:
            continue
        cols = [e for e in ents if e in user_item_df.columns]
        if not cols:
            members = []
        else:
            # compute for each user how many of these cols they have
            counts = user_item_df[cols].sum(axis=1)
            required = math.ceil(len(cols) * assignment_fraction - 1e-9)
            mask = counts >= required
            members = user_item_df.index[mask].tolist()
        key = tuple(sorted(ents))
        role_members[key] = members
        for uid in members:
            user_roles[uid].append(key)
    return role_members, user_roles
