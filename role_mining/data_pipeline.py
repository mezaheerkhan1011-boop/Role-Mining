# role_mining/data_pipeline.py
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

# Acceptable entitlement-like column names (case-insensitive)
_POSSIBLE_ENT_COLS = {
    "entitlements", "all_entitlements", "roles_permissions", "permissions",
    "accesses", "grants", "ent_list", "entitlement_list", "entitlement",
    "all_permissions", "all_accesses", "all_entitlements"
}

def _find_ent_col(df: pd.DataFrame):
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in _POSSIBLE_ENT_COLS:
        if candidate in cols_lower:
            return cols_lower[candidate]
    # fallback: any column name containing 'ent' or 'perm' or 'access'
    for lower_name, orig in cols_lower.items():
        if "ent" in lower_name or "perm" in lower_name or "access" in lower_name:
            return orig
    return None

def load_csv(path, ent_col_hint=None, user_col="user_id", role_col="role_id"):
    """
    Loads CSV and returns DataFrame with columns ['user_id','ent_list'].
    - ent_col_hint: optional exact column name to use (e.g. "All_Entitlements")
    - Parser supports JSON-like lists, comma/semicolon-separated lists, and space-separated tokens like "Read:DB View:Reports"
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, dtype=str)

    # drop role_col if present
    if role_col in df.columns:
        try:
            df = df.drop(columns=[role_col])
        except Exception:
            pass

    # decide entitlement column to use
    ent_col = None
    if ent_col_hint and ent_col_hint in df.columns:
        ent_col = ent_col_hint
    else:
        ent_col = _find_ent_col(df)

    if ent_col is None:
        raise ValueError("CSV must contain an entitlement-like column (e.g. 'entitlements', 'All_Entitlements', 'roles_permissions').")

    # normalize user_id column
    if user_col in df.columns:
        if user_col != "user_id":
            df = df.rename(columns={user_col: "user_id"})
    elif "EmpID" in df.columns:
        df = df.rename(columns={"EmpID": "user_id"})
    elif "EmployeeID" in df.columns:
        df = df.rename(columns={"EmployeeID": "user_id"})
    else:
        df = df.reset_index().rename(columns={"index": "user_id"})
        df["user_id"] = df["user_id"].astype(str)

    # parser for entitlement text -> list
    def parse_ents(x):
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        s = str(x).strip()
        # try json list
        if (s.startswith("[") and s.endswith("]")) or (s.startswith('"[') and s.endswith(']"')):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(i).strip() for i in parsed if str(i).strip()]
            except Exception:
                pass
        # clean surrounding chars
        for ch in ["[", "]", "(", ")", "{", "}", '"', "'"]:
            s = s.replace(ch, "")
        # choose separator: prefer comma, then semicolon
        sep = "," if "," in s else (";" if ";" in s else None)
        if sep:
            items = [i.strip() for i in s.split(sep) if i.strip()]
            return items
        # if tokens look like "Read:FinanceDB View:Reports" split on whitespace
        if " " in s and ":" in s:
            parts = [i.strip() for i in s.split() if i.strip()]
            return parts
        # fallback single item
        return [s] if s else []

    df["ent_list"] = df[ent_col].apply(parse_ents)

    # ensure user_id exists
    if "user_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "user_id"})
        df["user_id"] = df["user_id"].astype(str)

    return df[["user_id", "ent_list"]]

def build_user_item_matrix(df):
    exploded = df.explode("ent_list")
    exploded = exploded[exploded["ent_list"].notna()]
    pivot = pd.crosstab(exploded["user_id"], exploded["ent_list"])
    pivot = pivot.applymap(lambda x: 1 if x > 0 else 0)
    return pivot

def train_test_split_matrix(user_item_df, test_size=0.2, random_state=42):
    users = list(user_item_df.index)
    if len(users) < 2:
        return user_item_df.copy(), user_item_df.copy()
    train_users, test_users = train_test_split(users, test_size=test_size, random_state=random_state)
    return user_item_df.loc[train_users], user_item_df.loc[test_users]
