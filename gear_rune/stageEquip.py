# app.py
import streamlit as st
import pandas as pd
import math
from typing import List, Tuple

# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="Stage별 장비/룬 사용 빈도", layout="wide")
st.title("Stage별 장비/룬 사용 빈도")
st.caption("입력: 장비 로그 3개(stage_id, equip_id, use_count) + 룬 로그 3개(stage_id, rune_id, use_count) + 마스터 1개(id, type, name, grade)")

# =========================================================
# Constants
# =========================================================
REQUIRED_EQUIP_COLS = ["stage_id", "equip_id", "use_count"]
REQUIRED_RUNE_COLS = ["stage_id", "rune_id", "use_count"]
REQUIRED_MASTER_COLS = ["id", "type", "name", "grade"]

# =========================================================
# Utils
# =========================================================
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
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

def read_csvs(uploaded_files: List) -> pd.DataFrame:
    dfs = []
    for f in uploaded_files:
        df = pd.read_csv(f)
        df = _normalize_cols(df)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def validate_cols(df: pd.DataFrame, required: List[str], name: str) -> Tuple[bool, str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"[{name}] 필수 컬럼 누락: {missing}\n현재 컬럼: {list(df.columns)}"
    return True, ""

def safe_merge(master: pd.DataFrame, usage: pd.DataFrame, left_id_col: str, master_label: str) -> pd.DataFrame:
    merged = usage.merge(
        master,
        how="left",
        left_on=left_id_col,
        right_on="id",
        suffixes=("", "_master"),
    )
    merged["_master_missing"] = merged["name"].isna()
    missing_cnt = int(merged["_master_missing"].sum())
    if missing_cnt > 0:
        st.warning(f"{master_label}: 마스터 조인 실패 {missing_cnt}건 (id 미매칭). name/grade/type이 비어있을 수 있어요.")
    return merged

# =========================================================
# Compute
# =========================================================
def top_equips_by_type_per_stage(equip_usage: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """
    stage_id별 'type'별로 가장 많이 사용된 장비 1개씩 → stage당 최대 6개(타입 6종)
    """
    u = equip_usage.copy()
    u["use_count"] = _to_int_series(u["use_count"])
    u["equip_id"] = u["equip_id"].astype(str).str.strip()
    u["stage_id"] = u["stage_id"].astype(str).str.strip()

    agg = (
        u.groupby(["stage_id", "equip_id"], as_index=False)["use_count"]
         .sum()
    )

    m = master.copy()
    m["id"] = m["id"].astype(str).str.strip()

    merged = safe_merge(m, agg, "equip_id", "장비")
    merged["type"] = merged["type"].fillna("Unknown").astype(str)

    # stage_id, type별 use_count 내림차순 1등
    merged = merged.sort_values(["stage_id", "type", "use_count"], ascending=[True, True, False])
    top1 = merged.groupby(["stage_id", "type"], as_index=False).head(1)

    out = top1[["stage_id", "type", "equip_id", "name", "grade", "use_count"]].copy()
    out = out.rename(columns={"equip_id": "id", "use_count": "use_count_sum"})
    out = out.sort_values(["stage_id", "type"])
    return out

def top_runes_per_stage(rune_usage: pd.DataFrame, master: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """
    stage_id별 사용량 Top N 룬 (룬은 type 없이)
    """
    u = rune_usage.copy()
    u["use_count"] = _to_int_series(u["use_count"])
    u["rune_id"] = u["rune_id"].astype(str).str.strip()
    u["stage_id"] = u["stage_id"].astype(str).str.strip()

    agg = (
        u.groupby(["stage_id", "rune_id"], as_index=False)["use_count"]
         .sum()
    )

    m = master.copy()
    m["id"] = m["id"].astype(str).str.strip()

    merged = safe_merge(m, agg, "rune_id", "룬")
    merged = merged.sort_values(["stage_id", "use_count"], ascending=[True, False])

    topn = merged.groupby("stage_id", as_index=False).head(top_n)

    out = topn[["stage_id", "rune_id", "name", "grade", "use_count"]].copy()
    out = out.rename(columns={"rune_id": "id", "use_count": "use_count_sum"})
    return out

def stage_pivot_view(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    stage별로 Top 1~6을 가로로 보여주는 뷰
    - equip: type 기준 정렬(타입 6개)
    - rune : use_count 내림차순
    """
    if df.empty:
        return df

    tmp = df.copy()
    if kind == "equip":
        tmp = tmp.sort_values(["stage_id", "type"])
    else:
        tmp = tmp.sort_values(["stage_id", "use_count_sum"], ascending=[True, False])

    tmp["rank"] = tmp.groupby("stage_id").cumcount() + 1

    if kind == "equip":
        tmp["label"] = tmp.apply(
            lambda r: f"{r.get('type','')}: {r.get('name','')} (grade={r.get('grade','')}, id={r.get('id','')}, cnt={r.get('use_count_sum','')})",
            axis=1
        )
    else:
        tmp["label"] = tmp.apply(
            lambda r: f"{r.get('name','')} (grade={r.get('grade','')}, id={r.get('id','')}, cnt={r.get('use_count_sum','')})",
            axis=1
        )

    pv = tmp.pivot_table(index="stage_id", columns="rank", values="label", aggfunc="first")
    pv.columns = [f"Top {int(c)}" for c in pv.columns]
    pv = pv.reset_index()
    return pv

# =========================================================
# UI: Upload
# =========================================================
st.subheader("1) CSV 업로드")

colA, colB, colC = st.columns([1, 1, 1])

with colA:
    equip_files = st.file_uploader(
        "유저 장비 사용 로그 CSV 3개 업로드",
        type=["csv"],
        accept_multiple_files=True
    )
    st.caption("필수 컬럼: stage_id, equip_id, use_count")

with colB:
    rune_files = st.file_uploader(
        "유저 룬 사용 로그 CSV 3개 업로드",
        type=["csv"],
        accept_multiple_files=True
    )
    st.caption("필수 컬럼: stage_id, rune_id, use_count")

with colC:
    master_file = st.file_uploader(
        "장비/룬 마스터 CSV 1개 업로드",
        type=["csv"],
        accept_multiple_files=False
    )
    st.caption("필수 컬럼: id, type, name, grade")

st.divider()

if not equip_files or not rune_files or not master_file:
    st.info("장비 로그 3개, 룬 로그 3개, 마스터 1개를 모두 업로드하면 결과가 표시됩니다.")
    st.stop()

# =========================================================
# Load & Validate
# =========================================================
equip_usage = read_csvs(equip_files)
rune_usage = read_csvs(rune_files)
master = pd.read_csv(master_file)
master = _normalize_cols(master)

ok, msg = validate_cols(equip_usage, REQUIRED_EQUIP_COLS, "장비 로그(3개 합친 결과)")
if not ok:
    st.error(msg)
    st.stop()

ok, msg = validate_cols(rune_usage, REQUIRED_RUNE_COLS, "룬 로그(3개 합친 결과)")
if not ok:
    st.error(msg)
    st.stop()

ok, msg = validate_cols(master, REQUIRED_MASTER_COLS, "마스터")
if not ok:
    st.error(msg)
    st.stop()

# master normalize
master["id"] = master["id"].astype(str).str.strip()

# =========================================================
# Compute
# =========================================================
equip_top = top_equips_by_type_per_stage(equip_usage, master)
rune_top = top_runes_per_stage(rune_usage, master, top_n=6)

equip_pv = stage_pivot_view(equip_top, kind="equip")
rune_pv = stage_pivot_view(rune_top, kind="rune")

# =========================================================
# Output
# =========================================================
st.subheader("2) 결과")

tab1, tab2, tab3 = st.tabs(["Stage별 요약(가로)", "Stage 상세(세로)", "다운로드"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Stage별 Top 장비 6개 (타입별 1개)")
        st.dataframe(equip_pv, use_container_width=True)
    with c2:
        st.markdown("### Stage별 Top 룬 6개")
        st.dataframe(rune_pv, use_container_width=True)

with tab2:
    stages = sorted(set(equip_top["stage_id"].astype(str)) | set(rune_top["stage_id"].astype(str)))
    if not stages:
        st.warning("계산된 stage가 없습니다. (use_count가 전부 0이거나 데이터가 비어있을 수 있어요.)")
    else:
        selected_stage = st.selectbox("Stage 선택", stages, index=0)

        left, right = st.columns(2)
        with left:
            st.markdown("### 장비 (타입별 Top 1)")
            view_e = equip_top[equip_top["stage_id"].astype(str) == str(selected_stage)].copy()
            if not view_e.empty:
                view_e = view_e.sort_values(["type"])
            st.dataframe(view_e, use_container_width=True)

        with right:
            st.markdown("### 룬 (Top 6)")
            view_r = rune_top[rune_top["stage_id"].astype(str) == str(selected_stage)].copy()
            if not view_r.empty:
                view_r = view_r.sort_values(["use_count_sum"], ascending=False)
            st.dataframe(view_r, use_container_width=True)

with tab3:
    st.markdown("### 결과 CSV 다운로드")
    e_csv = equip_top.to_csv(index=False).encode("utf-8-sig")
    r_csv = rune_top.to_csv(index=False).encode("utf-8-sig")
    epv_csv = equip_pv.to_csv(index=False).encode("utf-8-sig")
    rpv_csv = rune_pv.to_csv(index=False).encode("utf-8-sig")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button("장비 결과(세로) 다운로드", e_csv, file_name="equip_top_vertical.csv", mime="text/csv")
    with col2:
        st.download_button("룬 결과(세로) 다운로드", r_csv, file_name="rune_top_vertical.csv", mime="text/csv")
    with col3:
        st.download_button("장비 요약(가로) 다운로드", epv_csv, file_name="equip_top_pivot.csv", mime="text/csv")
    with col4:
        st.download_button("룬 요약(가로) 다운로드", rpv_csv, file_name="rune_top_pivot.csv", mime="text/csv")

st.caption("장비: stage×type별 use_count 합산 후 1등(타입별 1개). 룬: stage별 use_count 합산 Top 6.")
