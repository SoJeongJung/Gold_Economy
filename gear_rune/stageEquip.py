# app.py
import streamlit as st
import pandas as pd
from typing import List, Tuple, Dict, Optional
import re

st.set_page_config(page_title="Stage별 장비/룬 사용 빈도 (세그먼트)", layout="wide")
st.title("Stage별 장비/룬 사용 빈도 (캐주얼/미드코어/해비)")

REQUIRED_EQUIP_COLS = ["stage_id", "equip_id", "use_count"]
REQUIRED_RUNE_COLS = ["stage_id", "rune_id", "use_count"]
REQUIRED_MASTER_COLS = ["id", "type", "name", "grade"]

SEGMENTS = [("casual", "캐주얼"), ("midcore", "미드코어"), ("heavy", "해비")]

EQUIP_TYPE_ORDER = ["hat", "neck", "ring", "coat", "belt", "boots"]

TYPE_CANON_MAP = {
    "hat": "hat",
    "neck": "neck",
    "ring": "ring",
    "coat": "coat",
    "belt": "belt",
    "boots": "boots",
}

# ---------------- Utils ----------------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def canon(c: str) -> str:
        c = str(c).strip().lower()
        c = re.sub(r"\s+", "", c)
        c = c.replace("-", "_")
        c = re.sub(r"_+", "_", c)
        return c

    df.columns = [canon(c) for c in df.columns]
    rename_map = {
        "equipid": "equip_id",
        "equipmentid": "equip_id",
        "runeid": "rune_id",
        "stageid": "stage_id",
        "usecount": "use_count",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    return df

def _to_int_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.strip()
         .replace({"nan": "0", "None": "0", "": "0"})
         .astype(float)
         .fillna(0)
         .astype(int)
    )

def validate_cols(df: pd.DataFrame, required: List[str], name: str) -> Tuple[bool, str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"[{name}] 필수 컬럼 누락: {missing}\n현재 컬럼: {list(df.columns)}"
    return True, ""

def sort_stage(df: pd.DataFrame, col: str = "stage") -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df
    tmp = df.copy()
    tmp["_stage_num"] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp.sort_values(by=["_stage_num", col], ascending=[True, True]).drop(columns=["_stage_num"])
    return tmp

def canon_type(x: str) -> str:
    if pd.isna(x):
        return "unknown"
    s = str(x).strip()
    s = re.sub(r"\s+", "", s).lower()
    return TYPE_CANON_MAP.get(s, s)

def normalize_id_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s = s.str.replace("\ufeff", "", regex=False)
    s = s.str.replace("\u200b", "", regex=False)
    s = s.str.strip()

    num = pd.to_numeric(s, errors="coerce")
    s2 = s.copy()
    mask = num.notna()
    if mask.any():
        s2.loc[mask] = num.loc[mask].astype("Int64").astype(str)
    return s2

def weighted_top_grade(df: pd.DataFrame, stage_col: str, grade_col: str, weight_col: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=str)
    g = (
        df.groupby([stage_col, grade_col], as_index=False)[weight_col]
          .sum()
          .sort_values([stage_col, weight_col], ascending=[True, False])
    )
    top = g.groupby(stage_col, as_index=False).head(1)
    return top.set_index(stage_col)[grade_col]

def safe_merge_with_debug(master: pd.DataFrame, usage: pd.DataFrame, left_id_col: str, master_label: str) -> pd.DataFrame:
    merged = usage.merge(master, how="left", left_on=left_id_col, right_on="id")
    missing_cnt = int(merged["name"].isna().sum())
    if missing_cnt > 0:
        st.warning(f"{master_label}: 마스터 조인 실패 {missing_cnt}건 (id 미매칭).")
    return merged

def format_name_count(name, cnt) -> str:
    if pd.isna(name) or name is None or str(name).strip() == "" or str(name) == "nan":
        return "누락"
    try:
        if pd.isna(cnt):
            return str(name)
        c = int(cnt)
        return f"{name} ({c})"
    except Exception:
        return f"{name} ({cnt})"

# ---------------- Compute Equip ----------------
def compute_equip_outputs(equip_usage: pd.DataFrame, master: pd.DataFrame):
    u = equip_usage.copy()
    u["use_count"] = _to_int_series(u["use_count"])
    u["stage_id"] = normalize_id_series(u["stage_id"])
    u["equip_id"] = normalize_id_series(u["equip_id"])

    agg = u.groupby(["stage_id", "equip_id"], as_index=False)["use_count"].sum()
    agg = agg.rename(columns={"use_count": "use_count_sum"})

    m = master.copy()
    m["id"] = normalize_id_series(m["id"])
    m["type"] = m["type"].apply(canon_type)
    m["name"] = m["name"].astype(str)
    m["grade"] = m["grade"].astype(str)

    merged = safe_merge_with_debug(m, agg, "equip_id", "장비")

    merged = merged.sort_values(["stage_id", "type", "use_count_sum"], ascending=[True, True, False])
    top1 = merged.groupby(["stage_id", "type"], as_index=False).head(1)

    top1_req = top1[top1["type"].isin(EQUIP_TYPE_ORDER)].copy()
    stage_grade = weighted_top_grade(top1_req, "stage_id", "grade", "use_count_sum")

    # wide name & count
    name_wide = top1_req.pivot_table(index="stage_id", columns="type", values="name", aggfunc="first")
    cnt_wide = top1_req.pivot_table(index="stage_id", columns="type", values="use_count_sum", aggfunc="first")

    for t in EQUIP_TYPE_ORDER:
        if t not in name_wide.columns:
            name_wide[t] = pd.NA
        if t not in cnt_wide.columns:
            cnt_wide[t] = pd.NA

    name_wide = name_wide[EQUIP_TYPE_ORDER]
    cnt_wide = cnt_wide[EQUIP_TYPE_ORDER]

    equip_final = name_wide.reset_index().rename(columns={"stage_id": "stage"})
    equip_final["grade"] = equip_final["stage"].map(stage_grade).fillna(pd.NA)
    equip_final = sort_stage(equip_final, "stage")

    equip_counts = cnt_wide.reset_index().rename(columns={"stage_id": "stage"})
    equip_counts["grade"] = equip_counts["stage"].map(stage_grade).fillna(pd.NA)
    equip_counts = sort_stage(equip_counts, "stage")

    equip_debug = top1[["stage_id", "type", "equip_id", "name", "grade", "use_count_sum"]].copy()
    equip_debug = equip_debug.rename(columns={"stage_id": "stage", "equip_id": "id"})
    equip_debug = sort_stage(equip_debug, "stage")

    return equip_final, equip_counts, equip_debug

# ---------------- Compute Rune ----------------
def compute_rune_outputs(rune_usage: pd.DataFrame, master: pd.DataFrame, top_n: int = 6):
    u = rune_usage.copy()
    u["use_count"] = _to_int_series(u["use_count"])
    u["stage_id"] = normalize_id_series(u["stage_id"])
    u["rune_id"] = normalize_id_series(u["rune_id"])

    agg = u.groupby(["stage_id", "rune_id"], as_index=False)["use_count"].sum()
    agg = agg.rename(columns={"use_count": "use_count_sum"})

    m = master.copy()
    m["id"] = normalize_id_series(m["id"])
    m["name"] = m["name"].astype(str)
    m["grade"] = m["grade"].astype(str)

    merged = agg.merge(m, how="left", left_on="rune_id", right_on="id")
    merged = merged.sort_values(["stage_id", "use_count_sum"], ascending=[True, False])

    topn = merged.groupby("stage_id", as_index=False).head(top_n).copy()
    topn["rank"] = topn.groupby("stage_id").cumcount() + 1

    stage_grade = weighted_top_grade(topn, "stage_id", "grade", "use_count_sum")

    name_wide = topn.pivot_table(index="stage_id", columns="rank", values="name", aggfunc="first")
    cnt_wide = topn.pivot_table(index="stage_id", columns="rank", values="use_count_sum", aggfunc="first")

    for r in range(1, top_n + 1):
        if r not in name_wide.columns:
            name_wide[r] = pd.NA
        if r not in cnt_wide.columns:
            cnt_wide[r] = pd.NA

    name_wide = name_wide[[r for r in range(1, top_n + 1)]]
    cnt_wide = cnt_wide[[r for r in range(1, top_n + 1)]]

    name_wide.columns = [f"Top{i}" for i in range(1, top_n + 1)]
    cnt_wide.columns = [f"Top{i}" for i in range(1, top_n + 1)]

    rune_final = name_wide.reset_index().rename(columns={"stage_id": "stage"})
    rune_final["grade"] = rune_final["stage"].map(stage_grade).fillna(pd.NA)
    rune_final = sort_stage(rune_final, "stage")

    rune_counts = cnt_wide.reset_index().rename(columns={"stage_id": "stage"})
    rune_counts["grade"] = rune_counts["stage"].map(stage_grade).fillna(pd.NA)
    rune_counts = sort_stage(rune_counts, "stage")

    rune_debug = topn[["stage_id", "rank", "rune_id", "name", "grade", "use_count_sum"]].copy()
    rune_debug = rune_debug.rename(columns={"stage_id": "stage", "rune_id": "id"})
    rune_debug = sort_stage(rune_debug, "stage")

    return rune_final, rune_counts, rune_debug

# =========================================================
# UI: Upload
# =========================================================
st.subheader("1) 마스터 업로드 (공용 1개)")
master_file = st.file_uploader("장비/룬 마스터 CSV", type=["csv"], accept_multiple_files=False)
st.caption("필수 컬럼: id, type, name, grade")

st.divider()
st.subheader("2) 세그먼트별 로그 업로드 (선택)")

cols = st.columns(3)
segment_uploads: Dict[str, Dict[str, Optional[object]]] = {}
for i, (seg_key, seg_name) in enumerate(SEGMENTS):
    with cols[i]:
        st.markdown(f"### {seg_name}")
        equip_f = st.file_uploader(f"{seg_name} - 장비 로그", type=["csv"], key=f"equip_{seg_key}")
        rune_f = st.file_uploader(f"{seg_name} - 룬 로그", type=["csv"], key=f"rune_{seg_key}")
        segment_uploads[seg_key] = {"name": seg_name, "equip": equip_f, "rune": rune_f}

if not master_file:
    st.info("마스터 CSV를 먼저 업로드하세요.")
    st.stop()

master = _normalize_cols(pd.read_csv(master_file))
ok, msg = validate_cols(master, REQUIRED_MASTER_COLS, "마스터")
if not ok:
    st.error(msg)
    st.stop()

available_segments = [k for k, v in segment_uploads.items() if (v["equip"] is not None or v["rune"] is not None)]
if not available_segments:
    st.info("세그먼트 중 최소 1개는 장비 또는 룬 로그를 업로드하세요.")
    st.stop()

tabs = st.tabs([segment_uploads[k]["name"] for k in available_segments])

def render_segment(seg_key: str, seg_name: str, equip_file, rune_file):
    st.markdown(f"## {seg_name}")

    equip_usage = None
    rune_usage = None

    if equip_file is not None:
        equip_usage = _normalize_cols(pd.read_csv(equip_file))
        ok, msg = validate_cols(equip_usage, REQUIRED_EQUIP_COLS, f"{seg_name} - 장비 로그")
        if not ok:
            st.error(msg)
            equip_usage = None

    if rune_file is not None:
        rune_usage = _normalize_cols(pd.read_csv(rune_file))
        ok, msg = validate_cols(rune_usage, REQUIRED_RUNE_COLS, f"{seg_name} - 룬 로그")
        if not ok:
            st.error(msg)
            rune_usage = None

    if equip_usage is None and rune_usage is None:
        st.warning("이 세그먼트는 처리 가능한 로그가 없습니다.")
        return

    equip_final = equip_counts = equip_debug = pd.DataFrame()
    rune_final = rune_counts = rune_debug = pd.DataFrame()

    if equip_usage is not None:
        equip_final, equip_counts, equip_debug = compute_equip_outputs(equip_usage, master)

    if rune_usage is not None:
        rune_final, rune_counts, rune_debug = compute_rune_outputs(rune_usage, master, top_n=6)

    # 누락 표기 / 카운트 0 처리
    if not equip_final.empty:
        equip_final = equip_final.fillna("누락")
    if not rune_final.empty:
        rune_final = rune_final.fillna("누락")

    if not equip_counts.empty:
        equip_counts = equip_counts.fillna(0)
    if not rune_counts.empty:
        rune_counts = rune_counts.fillna(0)

    # ✅ 요약용 "이름 (횟수)" 테이블 생성 (UI 깔끔)
    equip_display = pd.DataFrame()
    if not equip_final.empty and not equip_counts.empty:
        equip_display = equip_final.copy()
        for t in EQUIP_TYPE_ORDER:
            equip_display[t] = [
                format_name_count(n, c)
                for n, c in zip(equip_final[t].values, equip_counts[t].values)
            ]
        # grade는 그대로(가장 많이 사용된 grade)
        equip_display["grade"] = equip_final["grade"]

    rune_display = pd.DataFrame()
    top_cols = [f"Top{i}" for i in range(1, 7)]
    if not rune_final.empty and not rune_counts.empty:
        rune_display = rune_final.copy()
        for c in top_cols:
            rune_display[c] = [
                format_name_count(n, cnt)
                for n, cnt in zip(rune_final[c].values, rune_counts[c].values)
            ]
        rune_display["grade"] = rune_final["grade"]

    t1, t2 = st.tabs(["요약(이름+횟수)", "사용횟수/디버그"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 장비 요약 (장비명 (횟수))")
            cols_e = ["stage"] + EQUIP_TYPE_ORDER + ["grade"]
            if equip_display.empty:
                st.info("장비 결과가 없습니다.")
            else:
                st.dataframe(equip_display[cols_e], use_container_width=True)

        with c2:
            st.markdown("### 룬 요약 (룬명 (횟수))")
            cols_r = ["stage"] + top_cols + ["grade"]
            if rune_display.empty:
                st.info("룬 결과가 없습니다.")
            else:
                st.dataframe(rune_display[cols_r], use_container_width=True)

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 장비 use_count_sum (숫자)")
            cols_e = ["stage"] + EQUIP_TYPE_ORDER + ["grade"]
            if equip_counts.empty:
                st.info("장비 카운트 결과가 없습니다.")
            else:
                st.dataframe(equip_counts[cols_e], use_container_width=True)

        with c2:
            st.markdown("### 룬 use_count_sum (숫자)")
            cols_r = ["stage"] + top_cols + ["grade"]
            if rune_counts.empty:
                st.info("룬 카운트 결과가 없습니다.")
            else:
                st.dataframe(rune_counts[cols_r], use_container_width=True)

        st.divider()
        st.markdown("### 디버그(세로)")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### 장비 디버그")
            if equip_debug.empty:
                st.info("장비 디버그가 없습니다.")
            else:
                st.dataframe(equip_debug, use_container_width=True)
        with c4:
            st.markdown("#### 룬 디버그")
            if rune_debug.empty:
                st.info("룬 디버그가 없습니다.")
            else:
                st.dataframe(rune_debug, use_container_width=True)

for i, seg_key in enumerate(available_segments):
    with tabs[i]:
        info = segment_uploads[seg_key]
        render_segment(seg_key, info["name"], info["equip"], info["rune"])
