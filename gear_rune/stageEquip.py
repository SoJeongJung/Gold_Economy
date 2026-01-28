# app.py
import streamlit as st
import pandas as pd
from typing import List, Tuple, Dict, Optional
import re

# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="Stage별 장비/룬 사용 빈도 (세그먼트)", layout="wide")
st.title("Stage별 장비/룬 사용 빈도 (캐주얼/미드코어/해비)")

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

# 장비 타입 컬럼 순서(요구사항 고정)
EQUIP_TYPE_ORDER = ["hat", "neck", "ring", "coat", "belt"]

# =========================================================
# Utils
# =========================================================
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    안전장치: 공백/대소문자/하이픈 등 변형을 정리.
    (사용자가 공백 문제를 해결했더라도, 실수 재발 방지용)
    """
    df = df.copy()

    def canon(c: str) -> str:
        c = str(c).strip().lower()
        c = re.sub(r"\s+", "", c)      # 모든 공백 제거
        c = c.replace("-", "_")        # 하이픈 -> 언더스코어
        c = re.sub(r"_+", "_", c)      # 연속 언더스코어 정리
        return c

    df.columns = [canon(c) for c in df.columns]

    rename_map = {
        "equipid": "equip_id",
        "equipmentid": "equip_id",
        "equip_id": "equip_id",

        "runeid": "rune_id",
        "rune_id": "rune_id",

        "stageid": "stage_id",
        "stage_id": "stage_id",

        "usecount": "use_count",
        "use_count": "use_count",
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

def safe_merge(master: pd.DataFrame, usage: pd.DataFrame, left_id_col: str, master_label: str) -> pd.DataFrame:
    merged = usage.merge(
        master,
        how="left",
        left_on=left_id_col,
        right_on="id",
        suffixes=("", "_m"),
    )
    missing_cnt = int(merged["name"].isna().sum())
    if missing_cnt > 0:
        st.warning(f"{master_label}: 마스터 조인 실패 {missing_cnt}건 (id 미매칭). name/grade/type이 비어있을 수 있어요.")
    return merged

def weighted_top_grade(df: pd.DataFrame, stage_col: str, grade_col: str, weight_col: str) -> pd.Series:
    """
    stage별로 grade를 weight(use_count_sum)로 합산해서 가장 큰 grade를 선택
    """
    if df.empty:
        return pd.Series(dtype=str)

    g = (
        df.groupby([stage_col, grade_col], as_index=False)[weight_col]
          .sum()
          .sort_values([stage_col, weight_col], ascending=[True, False])
    )
    top = g.groupby(stage_col, as_index=False).head(1)
    s = top.set_index(stage_col)[grade_col]
    return s

# =========================================================
# Compute (Equip)
# =========================================================
def compute_equip_outputs(equip_usage: pd.DataFrame, master: pd.DataFrame):
    """
    반환:
      - equip_final: stage, hat, neck, ring, coat, belt, grade   (이름만)
      - equip_counts: stage, hat, neck, ring, coat, belt, grade  (use_count_sum)
      - equip_debug_long: stage_id/type/id/name/grade/use_count_sum (세로 디버그/다운로드)
    """
    u = equip_usage.copy()
    u["use_count"] = _to_int_series(u["use_count"])
    u["stage_id"] = u["stage_id"].astype(str).str.strip()
    u["equip_id"] = u["equip_id"].astype(str).str.strip()

    agg = u.groupby(["stage_id", "equip_id"], as_index=False)["use_count"].sum()
    agg = agg.rename(columns={"use_count": "use_count_sum"})

    m = master.copy()
    m["id"] = m["id"].astype(str).str.strip()
    m["type"] = m["type"].astype(str).str.strip().str.lower()

    merged = safe_merge(m, agg, "equip_id", "장비")

    # stage×type별 가장 많이 사용된 equip 1개
    merged = merged.sort_values(["stage_id", "type", "use_count_sum"], ascending=[True, True, False])
    top1 = merged.groupby(["stage_id", "type"], as_index=False).head(1)

    # 세로 디버그
    equip_debug_long = top1[["stage_id", "type", "equip_id", "name", "grade", "use_count_sum"]].copy()
    equip_debug_long = equip_debug_long.rename(columns={"equip_id": "id"})

    # 요구사항 타입만 남기기(원치 않는 타입이 섞이면 여기서 걸러짐)
    top1_req = top1[top1["type"].isin(EQUIP_TYPE_ORDER)].copy()

    # stage별 grade(가장 많이 사용된 grade) 산출: 선택된 타입들만 대상으로 weight 합산 후 최빈(가중)
    stage_grade = weighted_top_grade(top1_req, "stage_id", "grade", "use_count_sum")

    # 이름 wide
    name_wide = top1_req.pivot_table(index="stage_id", columns="type", values="name", aggfunc="first")
    # count wide
    cnt_wide = top1_req.pivot_table(index="stage_id", columns="type", values="use_count_sum", aggfunc="first")

    # 컬럼 정렬/보장
    for t in EQUIP_TYPE_ORDER:
        if t not in name_wide.columns:
            name_wide[t] = pd.NA
        if t not in cnt_wide.columns:
            cnt_wide[t] = pd.NA

    name_wide = name_wide[EQUIP_TYPE_ORDER]
    cnt_wide = cnt_wide[EQUIP_TYPE_ORDER]

    equip_final = name_wide.reset_index().rename(columns={"stage_id": "stage"})
    equip_final["grade"] = equip_final["stage"].map(stage_grade).fillna(pd.NA)

    equip_counts = cnt_wide.reset_index().rename(columns={"stage_id": "stage"})
    equip_counts["grade"] = equip_counts["stage"].map(stage_grade).fillna(pd.NA)

    equip_final = sort_stage(equip_final, "stage")
    equip_counts = sort_stage(equip_counts, "stage")
    equip_debug_long = equip_debug_long.copy()
    equip_debug_long = equip_debug_long.rename(columns={"stage_id": "stage"})
    equip_debug_long = sort_stage(equip_debug_long, "stage")

    return equip_final, equip_counts, equip_debug_long

# =========================================================
# Compute (Rune)
# =========================================================
def compute_rune_outputs(rune_usage: pd.DataFrame, master: pd.DataFrame, top_n: int = 6):
    """
    반환:
      - rune_final: stage, Top1..Top6, grade (이름만)
      - rune_counts: stage, Top1..Top6, grade (use_count_sum)
      - rune_debug_long: stage_id/rank/id/name/grade/use_count_sum (세로 디버그/다운로드)
    """
    u = rune_usage.copy()
    u["use_count"] = _to_int_series(u["use_count"])
    u["stage_id"] = u["stage_id"].astype(str).str.strip()
    u["rune_id"] = u["rune_id"].astype(str).str.strip()

    agg = u.groupby(["stage_id", "rune_id"], as_index=False)["use_count"].sum()
    agg = agg.rename(columns={"use_count": "use_count_sum"})

    m = master.copy()
    m["id"] = m["id"].astype(str).str.strip()

    merged = safe_merge(m, agg, "rune_id", "룬")

    merged = merged.sort_values(["stage_id", "use_count_sum"], ascending=[True, False])
    topn = merged.groupby("stage_id", as_index=False).head(top_n).copy()
    topn["rank"] = topn.groupby("stage_id").cumcount() + 1

    # stage별 grade(가장 많이 사용된 grade): Top6만 대상으로 가중 합산
    stage_grade = weighted_top_grade(topn, "stage_id", "grade", "use_count_sum")

    # 이름 wide
    name_wide = topn.pivot_table(index="stage_id", columns="rank", values="name", aggfunc="first")
    # count wide
    cnt_wide = topn.pivot_table(index="stage_id", columns="rank", values="use_count_sum", aggfunc="first")

    # Top1..Top6 보장
    for r in range(1, top_n + 1):
        if r not in name_wide.columns:
            name_wide[r] = pd.NA
        if r not in cnt_wide.columns:
            cnt_wide[r] = pd.NA

    name_wide = name_wide[[r for r in range(1, top_n + 1)]]
    cnt_wide = cnt_wide[[r for r in range(1, top_n + 1)]]

    name_wide.columns = [f"Top{r}" for r in range(1, top_n + 1)]
    cnt_wide.columns = [f"Top{r}" for r in range(1, top_n + 1)]

    rune_final = name_wide.reset_index().rename(columns={"stage_id": "stage"})
    rune_final["grade"] = rune_final["stage"].map(stage_grade).fillna(pd.NA)

    rune_counts = cnt_wide.reset_index().rename(columns={"stage_id": "stage"})
    rune_counts["grade"] = rune_counts["stage"].map(stage_grade).fillna(pd.NA)

    rune_final = sort_stage(rune_final, "stage")
    rune_counts = sort_stage(rune_counts, "stage")

    rune_debug_long = topn[["stage_id", "rank", "rune_id", "name", "grade", "use_count_sum"]].copy()
    rune_debug_long = rune_debug_long.rename(columns={"stage_id": "stage", "rune_id": "id"})
    rune_debug_long = sort_stage(rune_debug_long, "stage")

    return rune_final, rune_counts, rune_debug_long

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
            key=f"equip_{seg_key}",
        )
        st.caption("장비 로그 컬럼: stage_id, equip_id, use_count")

        rune_f = st.file_uploader(
            f"{seg_name} - 룬 로그",
            type=["csv"],
            accept_multiple_files=False,
            key=f"rune_{seg_key}",
        )
        st.caption("룬 로그 컬럼: stage_id, rune_id, use_count")

        segment_uploads[seg_key] = {"name": seg_name, "equip": equip_f, "rune": rune_f}

st.divider()

if not master_file:
    st.info("먼저 마스터 CSV를 업로드하세요.")
    st.stop()

master = _normalize_cols(pd.read_csv(master_file))
ok, msg = validate_cols(master, REQUIRED_MASTER_COLS, "마스터")
if not ok:
    st.error(msg)
    st.stop()

# master normalize
master["id"] = master["id"].astype(str).str.strip()
master["type"] = master["type"].astype(str).str.strip().str.lower()
master["name"] = master["name"].astype(str)
master["grade"] = master["grade"].astype(str)

available_segments = []
for seg_key, info in segment_uploads.items():
    if info["equip"] is not None or info["rune"] is not None:
        available_segments.append(seg_key)

if not available_segments:
    st.info("캐주얼/미드코어/해비 중 최소 1개 세그먼트의 장비 또는 룬 로그를 업로드하세요.")
    st.stop()

# =========================================================
# Render per segment
# =========================================================
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

    # Compute
    equip_final = pd.DataFrame()
    equip_counts = pd.DataFrame()
    equip_debug = pd.DataFrame()

    rune_final = pd.DataFrame()
    rune_counts = pd.DataFrame()
    rune_debug = pd.DataFrame()

    if equip_usage is not None:
        equip_final, equip_counts, equip_debug = compute_equip_outputs(equip_usage, master)

    if rune_usage is not None:
        rune_final, rune_counts, rune_debug = compute_rune_outputs(rune_usage, master, top_n=6)

    # Tabs
    t1, t2, t3 = st.tabs(["요약 테이블(요구 포맷)", "사용횟수/디버그", "다운로드"])

    with t1:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 장비 (stage, hat, neck, ring, coat, belt, grade)")
            if equip_final.empty:
                st.info("장비 로그가 업로드되지 않았거나, 계산 결과가 없습니다.")
            else:
                # 요구 컬럼 순서 보장
                cols = ["stage"] + EQUIP_TYPE_ORDER + ["grade"]
                for col in cols:
                    if col not in equip_final.columns:
                        equip_final[col] = pd.NA
                st.dataframe(equip_final[cols], use_container_width=True)

        with c2:
            st.markdown("### 룬 (stage, Top1..Top6, grade)")
            if rune_final.empty:
                st.info("룬 로그가 업로드되지 않았거나, 계산 결과가 없습니다.")
            else:
                cols = ["stage"] + [f"Top{i}" for i in range(1, 7)] + ["grade"]
                for col in cols:
                    if col not in rune_final.columns:
                        rune_final[col] = pd.NA
                st.dataframe(rune_final[cols], use_container_width=True)

    with t2:
        st.markdown("### 사용 횟수(use_count_sum) 테이블")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 장비 사용 횟수 (stage, hat..belt, grade)")
            if equip_counts.empty:
                st.info("장비 사용횟수 테이블이 없습니다.")
            else:
                cols = ["stage"] + EQUIP_TYPE_ORDER + ["grade"]
                for col in cols:
                    if col not in equip_counts.columns:
                        equip_counts[col] = pd.NA
                st.dataframe(equip_counts[cols], use_container_width=True)

        with c2:
            st.markdown("#### 룬 사용 횟수 (stage, Top1..Top6, grade)")
            if rune_counts.empty:
                st.info("룬 사용횟수 테이블이 없습니다.")
            else:
                cols = ["stage"] + [f"Top{i}" for i in range(1, 7)] + ["grade"]
                for col in cols:
                    if col not in rune_counts.columns:
                        rune_counts[col] = pd.NA
                st.dataframe(rune_counts[cols], use_container_width=True)

        st.divider()
        st.markdown("### 디버그(세로) - 조인/선정 결과 확인용")
        c3, c4 = st.columns(2)

        with c3:
            st.markdown("#### 장비: stage×type별 Top1 (name/grade/use_count_sum)")
            if equip_debug.empty:
                st.info("장비 디버그가 없습니다.")
            else:
                st.dataframe(equip_debug, use_container_width=True)

        with c4:
            st.markdown("#### 룬: stage별 Top6 (rank/name/grade/use_count_sum)")
            if rune_debug.empty:
                st.info("룬 디버그가 없습니다.")
            else:
                st.dataframe(rune_debug, use_container_width=True)

    with t3:
        st.markdown("### CSV 다운로드")
        dl_cols = st.columns(6)

        # Equip
        if not equip_final.empty:
            cols = ["stage"] + EQUIP_TYPE_ORDER + ["grade"]
            equip_final_csv = equip_final[cols].to_csv(index=False).encode("utf-8-sig")
            with dl_cols[0]:
                st.download_button(
                    "장비 요약(요구포맷)",
                    equip_final_csv,
                    file_name=f"{seg_key}_equip_summary.csv",
                    mime="text/csv",
                    key=f"{seg_key}_dl_equip_summary",
                )
        if not equip_counts.empty:
            cols = ["stage"] + EQUIP_TYPE_ORDER + ["grade"]
            equip_counts_csv = equip_counts[cols].to_csv(index=False).encode("utf-8-sig")
            with dl_cols[1]:
                st.download_button(
                    "장비 사용횟수",
                    equip_counts_csv,
                    file_name=f"{seg_key}_equip_counts.csv",
                    mime="text/csv",
                    key=f"{seg_key}_dl_equip_counts",
                )
        if not equip_debug.empty:
            equip_debug_csv = equip_debug.to_csv(index=False).encode("utf-8-sig")
            with dl_cols[2]:
                st.download_button(
                    "장비 디버그(세로)",
                    equip_debug_csv,
                    file_name=f"{seg_key}_equip_debug_long.csv",
                    mime="text/csv",
                    key=f"{seg_key}_dl_equip_debug",
                )

        # Rune
        if not rune_final.empty:
            cols = ["stage"] + [f"Top{i}" for i in range(1, 7)] + ["grade"]
            rune_final_csv = rune_final[cols].to_csv(index=False).encode("utf-8-sig")
            with dl_cols[3]:
                st.download_button(
                    "룬 요약(요구포맷)",
                    rune_final_csv,
                    file_name=f"{seg_key}_rune_summary.csv",
                    mime="text/csv",
                    key=f"{seg_key}_dl_rune_summary",
                )
        if not rune_counts.empty:
            cols = ["stage"] + [f"Top{i}" for i in range(1, 7)] + ["grade"]
            rune_counts_csv = rune_counts[cols].to_csv(index=False).encode("utf-8-sig")
            with dl_cols[4]:
                st.download_button(
                    "룬 사용횟수",
                    rune_counts_csv,
                    file_name=f"{seg_key}_rune_counts.csv",
                    mime="text/csv",
                    key=f"{seg_key}_dl_rune_counts",
                )
        if not rune_debug.empty:
            rune_debug_csv = rune_debug.to_csv(index=False).encode("utf-8-sig")
            with dl_cols[5]:
                st.download_button(
                    "룬 디버그(세로)",
                    rune_debug_csv,
                    file_name=f"{seg_key}_rune_debug_long.csv",
                    mime="text/csv",
                    key=f"{seg_key}_dl_rune_debug",
                )

    st.caption(
        "장비 grade: (선택된 타입들에 대해) grade별 use_count_sum을 합산하여 가장 큰 grade를 표시합니다. "
        "룬 grade: Top6 내 grade별 use_count_sum 합산 1등 grade를 표시합니다. "
        "스테이지(stage)는 숫자 기준 오름차순으로 정렬됩니다."
    )

# =========================================================
# Top-level segment tabs (uploaded only)
# =========================================================
tab_labels = [segment_uploads[k]["name"] for k in available_segments]
tabs = st.tabs(tab_labels)

for idx, seg_key in enumerate(available_segments):
    info = segment_uploads[seg_key]
    with tabs[idx]:
        render_segment(seg_key, info["name"], info["equip"], info["rune"])
