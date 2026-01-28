# app.py
import streamlit as st
import pandas as pd
from typing import List, Tuple, Dict, Optional

# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="Stage별 장비/룬 사용 빈도 (세그먼트)", layout="wide")
st.title("Stage별 장비/룬 사용 빈도 (캐주얼/미드코어/해비)")
st.caption("세그먼트별 로그를 각각 업로드하면 업로드된 세그먼트만 탭으로 표시됩니다.")

# =========================================================
# Constants
# =========================================================
REQUIRED_EQUIP_COLS = ["stage_id", "equip_id", "use_count"]
REQUIRED_RUNE_COLS = ["stage_id", "rune_id", "use_count"]
REQUIRED_MASTER_COLS = ["id", "type", "name", "grade"]

SEGMENTS = [
    ("casual", "캐주얼"),
    ("midcore", "미드코어"),
    ("heavy", "해비"),
]

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

    merged = merged.sort_values(["stage_id", "type", "use_count"], ascending=[True, True, False])
    top1 = merged.groupby(["stage_id", "type"], as_index=False).head(1)

    out = top1[["stage_id", "type", "equip_id", "name", "grade", "use_count"]].copy()
    out = out.rename(columns={"equip_id": "id", "use_count": "use_count_sum"})
    out = out.sort_values(["stage_id", "type"])
    return out

def top_runes_per_stage(rune_usage: pd.DataFrame, master: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """
    stage_id별 사용량 Top N 룬
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
    if df is None or df.empty:
        return pd.DataFrame()

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
st.subheader("1) 마스터 업로드 (공용 1개)")
master_file = st.file_uploader("장비/룬 마스터 CSV", type=["csv"], accept_multiple_files=False)
st.caption("필수 컬럼: id, type, name, grade")

st.divider()
st.subheader("2) 세그먼트별 로그 업로드 (선택 업로드 가능)")

upload_cols = st.columns(3)
segment_uploads: Dict[str, Dict[str, Optional[object]]] = {}

for i, (seg_key, seg_name) in enumerate(SEGMENTS):
    with upload_cols[i]:
        st.markdown(f"### {seg_name}")
        equip_f = st.file_uploader(
            f"{seg_name} - 장비 로그",
            type=["csv"],
            accept_multiple_files=False,
            key=f"equip_{seg_key}"
        )
        st.caption("장비 로그 컬럼: stage_id, equip_id, use_count")

        rune_f = st.file_uploader(
            f"{seg_name} - 룬 로그",
            type=["csv"],
            accept_multiple_files=False,
            key=f"rune_{seg_key}"
        )
        st.caption("룬 로그 컬럼: stage_id, rune_id, use_count")

        segment_uploads[seg_key] = {"name": seg_name, "equip": equip_f, "rune": rune_f}

st.divider()

if not master_file:
    st.info("먼저 마스터 CSV를 업로드하세요.")
    st.stop()

# master load/validate
master = pd.read_csv(master_file)
master = _normalize_cols(master)

ok, msg = validate_cols(master, REQUIRED_MASTER_COLS, "마스터")
if not ok:
    st.error(msg)
    st.stop()

master["id"] = master["id"].astype(str).str.strip()

# 어떤 세그먼트가 업로드됐는지 판단 (장비/룬 둘 중 하나라도 있으면 탭 생성)
available_segments = []
for seg_key, info in segment_uploads.items():
    if info["equip"] is not None or info["rune"] is not None:
        available_segments.append(seg_key)

if not available_segments:
    st.info("캐주얼/미드코어/해비 중 최소 1개 세그먼트의 장비 또는 룬 로그를 업로드하세요.")
    st.stop()

# =========================================================
# Per-segment rendering
# =========================================================
def _empty_equip_top() -> pd.DataFrame:
    return pd.DataFrame(columns=["stage_id", "type", "id", "name", "grade", "use_count_sum"])

def _empty_rune_top() -> pd.DataFrame:
    return pd.DataFrame(columns=["stage_id", "id", "name", "grade", "use_count_sum"])

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
        st.warning("이 세그먼트는 업로드된 파일이 없거나(혹은 컬럼 오류로) 처리할 수 없습니다.")
        return

    # ✅ 빈 DF라도 '컬럼을 가진' 형태로 초기화 (KeyError 방지)
    equip_top = _empty_equip_top()
    rune_top = _empty_rune_top()
    equip_pv = pd.DataFrame()
    rune_pv = pd.DataFrame()

    if equip_usage is not None:
        equip_top = top_equips_by_type_per_stage(equip_usage, master)
        equip_pv = stage_pivot_view(equip_top, kind="equip")

    if rune_usage is not None:
        rune_top = top_runes_per_stage(rune_usage, master, top_n=6)
        rune_pv = stage_pivot_view(rune_top, kind="rune")

    inner_tabs = st.tabs(["Stage별 요약(가로)", "Stage 상세(세로)", "다운로드"])

    with inner_tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Stage별 Top 장비 6개 (타입별 1개)")
            if equip_pv.empty:
                st.info("장비 로그가 업로드되지 않았거나, 계산 결과가 없습니다.")
            else:
                st.dataframe(equip_pv, use_container_width=True)

        with c2:
            st.markdown("### Stage별 Top 룬 6개")
            if rune_pv.empty:
                st.info("룬 로그가 업로드되지 않았거나, 계산 결과가 없습니다.")
            else:
                st.dataframe(rune_pv, use_container_width=True)

    with inner_tabs[1]:
        # ✅ stage union도 컬럼 존재/empty 안전하게
        equip_stages = set(equip_top["stage_id"].astype(str)) if ("stage_id" in equip_top.columns and not equip_top.empty) else set()
        rune_stages = set(rune_top["stage_id"].astype(str)) if ("stage_id" in rune_top.columns and not rune_top.empty) else set()
        stages = sorted(equip_stages | rune_stages)

        if not stages:
            st.warning("계산된 stage가 없습니다. (use_count가 전부 0이거나 데이터가 비어있을 수 있어요.)")
        else:
            selected_stage = st.selectbox("Stage 선택", stages, index=0, key=f"stage_{seg_key}")

            left, right = st.columns(2)
            with left:
                st.markdown("### 장비 (타입별 Top 1)")
                if equip_top.empty:
                    st.info("장비 결과가 없습니다.")
                else:
                    view_e = equip_top[equip_top["stage_id"].astype(str) == str(selected_stage)].copy()
                    if not view_e.empty:
                        view_e = view_e.sort_values(["type"])
                    st.dataframe(view_e, use_container_width=True)

            with right:
                st.markdown("### 룬 (Top 6)")
                if rune_top.empty:
                    st.info("룬 결과가 없습니다.")
                else:
                    view_r = rune_top[rune_top["stage_id"].astype(str) == str(selected_stage)].copy()
                    if not view_r.empty:
                        view_r = view_r.sort_values(["use_count_sum"], ascending=False)
                    st.dataframe(view_r, use_container_width=True)

    with inner_tabs[2]:
        st.markdown("### 결과 CSV 다운로드")
        col1, col2, col3, col4 = st.columns(4)

        if not equip_top.empty:
            e_csv = equip_top.to_csv(index=False).encode("utf-8-sig")
            epv_csv = equip_pv.to_csv(index=False).encode("utf-8-sig") if not equip_pv.empty else None
            with col1:
                st.download_button(
                    "장비 결과(세로) 다운로드",
                    e_csv,
                    file_name=f"{seg_key}_equip_top_vertical.csv",
                    mime="text/csv",
                    key=f"dl_e_v_{seg_key}"
                )
            with col3:
                if epv_csv is not None:
                    st.download_button(
                        "장비 요약(가로) 다운로드",
                        epv_csv,
                        file_name=f"{seg_key}_equip_top_pivot.csv",
                        mime="text/csv",
                        key=f"dl_e_p_{seg_key}"
                    )
        if not rune_top.empty:
            r_csv = rune_top.to_csv(index=False).encode("utf-8-sig")
            rpv_csv = rune_pv.to_csv(index=False).encode("utf-8-sig") if not rune_pv.empty else None
            with col2:
                st.download_button(
                    "룬 결과(세로) 다운로드",
                    r_csv,
                    file_name=f"{seg_key}_rune_top_vertical.csv",
                    mime="text/csv",
                    key=f"dl_r_v_{seg_key}"
                )
            with col4:
                if rpv_csv is not None:
                    st.download_button(
                        "룬 요약(가로) 다운로드",
                        rpv_csv,
                        file_name=f"{seg_key}_rune_top_pivot.csv",
                        mime="text/csv",
                        key=f"dl_r_p_{seg_key}"
                    )

    st.caption("장비: stage×type별 use_count 합산 후 1등(타입별 1개). 룬: stage별 use_count 합산 Top 6.")

# =========================================================
# Top-level segment tabs (only uploaded)
# =========================================================
tab_labels = [segment_uploads[k]["name"] for k in available_segments]
tabs = st.tabs(tab_labels)

for idx, seg_key in enumerate(available_segments):
    info = segment_uploads[seg_key]
    with tabs[idx]:
        render_segment(seg_key, info["name"], info["equip"], info["rune"])
