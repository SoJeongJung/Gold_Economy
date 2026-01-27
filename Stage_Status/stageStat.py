# stageStat.py
# Streamlit app: Stage clear combat power + power contributions (equip/rune/agency/equip-level)
#
# 이번 수정(요청 반영):
# 1) 룩업 CSV 1(Agency) 구조 변경
#    - 컬럼: agencyLv, agencyAtk, agencyHp
#    - agency_power = agencyAtk*4 + agencyHp 로 계산
#    - agencyAtk/agencyHp 를 스냅샷 row에 저장 (옵션 계산에 사용)
# 2) 옵션 계산(optionAtk/optionHp) 시 "agencyAtk/agencyHp"를 합산에 포함
#    - optionAtk_power_total = ( (equip_atk + rune_atk + agencyAtk) * sum(optionAtk%) ) * 4
#    - optionHp_power_total  = ( (equip_hp  + rune_hp  + agencyHp ) * sum(optionHp%) )
#
# 유지(기존 UI/구조 최대한 유지):
# - equips/runes: name 중심 표시 (id는 검증용 유지)
# - gap_total_minus_calc / gap_label
# - 표 칼럼 표시 설정(전체/유저별)
# - 탭 구성
# - gradeType 최빈(스테이지 평균)도 유지
#
# 실행:
#   python3 -m streamlit run stageStat.py

import json
import ast
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# =========================
# Page config
# =========================
st.set_page_config(page_title="Stage Status (Power Breakdown)", layout="wide")
st.title("Stage Status – 전투력/요소 기여 분석")


# =========================
# Helpers
# =========================
def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _to_float(x) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        if isinstance(x, float) and np.isnan(x):
            return 0.0
        return float(x)
    s = str(x).strip()
    if s == "":
        return 0.0
    try:
        return float(s.replace(",", ""))
    except Exception:
        return 0.0


def _to_int(x) -> int:
    try:
        return int(round(_to_float(x)))
    except Exception:
        return 0


def _safe_json_loads(s: Any) -> Any:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    if isinstance(s, (dict, list)):
        return s
    txt = str(s).strip()
    if txt == "":
        return None

    try:
        return json.loads(txt)
    except Exception:
        pass

    try:
        return ast.literal_eval(txt)
    except Exception:
        pass

    try:
        txt2 = txt.replace("'", '"')
        return json.loads(txt2)
    except Exception:
        return None


def _get_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    return None


def _to_ratio_percent(x) -> float:
    """
    퍼센트 입력을 안전하게 '비율(0~1)'로 변환.
    허용:
      - "5%"  -> 0.05
      - "5"   -> 0.05   (퍼센트로 간주)
      - 5     -> 0.05
      - 0.05  -> 0.05   (이미 비율이면 그대로)
      - ""/NaN -> 0.0
    """
    if x is None:
        return 0.0
    if isinstance(x, float) and np.isnan(x):
        return 0.0

    s = str(x).strip()
    if s == "":
        return 0.0

    has_pct = "%" in s
    s2 = s.replace("%", "").replace(",", "").strip()

    try:
        v = float(s2)
    except Exception:
        return 0.0

    if has_pct:
        return v / 100.0

    if 0.0 <= v <= 1.0:
        return v
    return v / 100.0


def _mode_from_list_series(series: pd.Series) -> str:
    vals: List[str] = []
    for x in series:
        if isinstance(x, list):
            vals.extend([str(v) for v in x if str(v).strip() != ""])
    if not vals:
        return ""
    c = Counter(vals)
    return c.most_common(1)[0][0]


# =========================
# Data models
# =========================
@dataclass
class ItemRow:
    id: str
    name: str
    slotType: str
    atkBase: float
    atkInc: float
    hpBase: float
    hpInc: float
    # ratio(0~1)
    optionAtk: float
    optionHp: float
    optionAtkBase: float
    optionHpBase: float
    gradeType: str


@dataclass
class AgencyRow:
    agencyLv: int
    agencyAtk: float
    agencyHp: float

    @property
    def agencyPower(self) -> float:
        return self.agencyAtk * 4.0 + self.agencyHp


# =========================
# Loaders
# =========================
def load_item_lookup(item_file) -> Dict[str, ItemRow]:
    df = pd.read_csv(item_file)
    df = _clean_cols(df)

    id_col = _get_col(df, "id")
    name_col = _get_col(df, "name")
    slot_col = _get_col(df, "slotType", "slottype", "slot_type")
    if id_col is None or name_col is None or slot_col is None:
        raise ValueError("Item lookup CSV must include: id, name, slotType, atkBase, atkInc, hpBase, hpInc")

    for c in ["atkBase", "atkInc", "hpBase", "hpInc"]:
        cc = _get_col(df, c)
        if cc is None:
            df[c] = 0
        else:
            df[c] = pd.to_numeric(df[cc], errors="coerce").fillna(0)

    # 옵션 컬럼 (없으면 0), ratio(0~1)로 통일
    for c in ["optionAtk", "optionHp", "optionAtkBase", "optionHpBase"]:
        cc = _get_col(df, c)
        if cc is None:
            df[c] = 0.0
        else:
            df[c] = df[cc].apply(_to_ratio_percent)

    # gradeType (없으면 "")
    grade_col = _get_col(df, "gradeType", "gradetype", "grade_type")
    if grade_col is None:
        df["gradeType"] = ""
    else:
        df["gradeType"] = df[grade_col].fillna("").astype(str)

    df["id_str"] = df[id_col].astype(str)

    out: Dict[str, ItemRow] = {}
    for _, r in df.iterrows():
        iid = str(r["id_str"])
        out[iid] = ItemRow(
            id=iid,
            name=str(r[name_col]),
            slotType=str(r[slot_col]),
            atkBase=float(r["atkBase"]),
            atkInc=float(r["atkInc"]),
            hpBase=float(r["hpBase"]),
            hpInc=float(r["hpInc"]),
            optionAtk=float(r["optionAtk"]),
            optionHp=float(r["optionHp"]),
            optionAtkBase=float(r["optionAtkBase"]),
            optionHpBase=float(r["optionHpBase"]),
            gradeType=str(r["gradeType"]),
        )
    return out


def load_agency_lookup(agency_file) -> Dict[int, AgencyRow]:
    """
    룩업 CSV 1 (Agency)
      - agencyLv, agencyAtk, agencyHp
    """
    df = pd.read_csv(agency_file)
    df = _clean_cols(df)

    lv_col = _get_col(df, "agencyLv", "agencylev", "agency_lv", "lvl", "level")
    atk_col = _get_col(df, "agencyAtk", "agency_atk", "atk")
    hp_col = _get_col(df, "agencyHp", "agency_hp", "hp")
    if lv_col is None or atk_col is None or hp_col is None:
        raise ValueError("Agency lookup CSV must include columns: agencyLv, agencyAtk, agencyHp")

    df["lv_i"] = df[lv_col].apply(_to_int)
    df["atk_f"] = df[atk_col].apply(_to_float)
    df["hp_f"] = df[hp_col].apply(_to_float)

    out: Dict[int, AgencyRow] = {}
    for _, r in df.iterrows():
        lv = int(r["lv_i"])
        out[lv] = AgencyRow(agencyLv=lv, agencyAtk=float(r["atk_f"]), agencyHp=float(r["hp_f"]))
    return out


# =========================
# Power calc
# =========================
def calc_item_power(item: ItemRow, slot_lv: int, is_equip: bool) -> Tuple[float, float, float]:
    """
    Returns (atk_calc, hp_calc, power)
    - Equip: atk = atkBase + atkInc*slot_lv, hp = hpBase + hpInc*slot_lv
    - Rune:  atk = atkBase, hp = hpBase (Inc ignored, slot_lv ignored)
    - power = atk*4 + hp
    """
    if is_equip:
        atk = item.atkBase + item.atkInc * slot_lv
        hp = item.hpBase + item.hpInc * slot_lv
    else:
        atk = item.atkBase
        hp = item.hpBase
    power = atk * 4.0 + hp
    return atk, hp, power


def sum_equip_rune_power(
    equip_ids: List[str],
    rune_ids: List[str],
    slot_lv_by_type_id: Dict[int, int],
    slotType_to_type_id: Dict[str, int],
    item_map: Dict[str, ItemRow],
    agency_atk: float,
    agency_hp: float,
) -> Tuple[
    float, float, float, float,
    float, float, float, float,
    float, float, float, float,
    List[str], List[str],
    List[str], List[str]
]:
    """
    Returns:
      equip_power, rune_power, equip_base_power, equip_level_power,
      optionAtk_power_total, optionHp_power_total, optionAtk_power_self, optionHp_power_self,
      equip_atk_sum, equip_hp_sum, rune_atk_sum, rune_hp_sum,
      equip_names, rune_names,
      equip_grades, rune_grades
    """
    equip_power = 0.0
    equip_base_power = 0.0
    rune_power = 0.0

    equip_names: List[str] = []
    rune_names: List[str] = []
    equip_grades: List[str] = []
    rune_grades: List[str] = []

    equip_atk_sum = 0.0
    equip_hp_sum = 0.0
    rune_atk_sum = 0.0
    rune_hp_sum = 0.0

    optionAtk_pct_sum = 0.0  # ratio(0~1) 합
    optionHp_pct_sum = 0.0

    optionAtk_power_self = 0.0
    optionHp_power_self = 0.0

    # ---- equips
    for eid in equip_ids:
        it = item_map.get(str(eid))
        if not it:
            equip_names.append(f"(missing:{eid})")
            equip_grades.append("")
            continue

        equip_names.append(it.name)
        equip_grades.append(it.gradeType or "")

        type_id = slotType_to_type_id.get(it.slotType)
        lv = slot_lv_by_type_id.get(int(type_id), 0) if type_id is not None else 0

        atk, hp, p = calc_item_power(it, lv, is_equip=True)
        _, _, base = calc_item_power(it, 0, is_equip=True)

        equip_power += p
        equip_base_power += base

        equip_atk_sum += atk
        equip_hp_sum += hp

        # 총합 기준 옵션% 누적(장비)
        optionAtk_pct_sum += it.optionAtk
        optionHp_pct_sum += it.optionHp

        # 장비 "자기 자신" 기준 옵션
        if it.optionAtkBase:
            optionAtk_power_self += (atk * it.optionAtkBase) * 4.0
        if it.optionHpBase:
            optionHp_power_self += (hp * it.optionHpBase)

    # ---- runes
    for rid in rune_ids:
        it = item_map.get(str(rid))
        if not it:
            rune_names.append(f"(missing:{rid})")
            rune_grades.append("")
            continue

        rune_names.append(it.name)
        rune_grades.append(it.gradeType or "")

        atk, hp, p = calc_item_power(it, 0, is_equip=False)
        rune_power += p

        rune_atk_sum += atk
        rune_hp_sum += hp

        # 총합 기준 옵션% 누적(룬)
        optionAtk_pct_sum += it.optionAtk
        optionHp_pct_sum += it.optionHp

    equip_level_power = equip_power - equip_base_power

    # ---- total option base (요청 반영: agencyAtk/agencyHp 포함)
    total_atk_pre_option = equip_atk_sum + rune_atk_sum + float(agency_atk)
    total_hp_pre_option = equip_hp_sum + rune_hp_sum + float(agency_hp)

    optionAtk_power_total = (total_atk_pre_option * optionAtk_pct_sum) * 4.0
    optionHp_power_total = (total_hp_pre_option * optionHp_pct_sum)

    return (
        equip_power,
        rune_power,
        equip_base_power,
        equip_level_power,
        optionAtk_power_total,
        optionHp_power_total,
        optionAtk_power_self,
        optionHp_power_self,
        equip_atk_sum,
        equip_hp_sum,
        rune_atk_sum,
        rune_hp_sum,
        equip_names,
        rune_names,
        equip_grades,
        rune_grades,
    )


# =========================
# Parse user CSV
# =========================
def parse_user_csv(user_file) -> pd.DataFrame:
    df = pd.read_csv(user_file)
    df = _clean_cols(df)

    ev_col = _get_col(df, "Event Name", "event", "event_name")
    time_col = _get_col(df, "Time", "time")
    did_col = _get_col(df, "Distinct ID", "distinct_id")
    stage_lv_col = _get_col(df, "stage_lv", "Stage Lv", "stage level", "stage_lv ")
    stage_id_col = _get_col(df, "stage_id", "Stage ID")
    combat_col = _get_col(df, "combat_power", "combat power", "combatPower", "combat_power ")
    game_state_col = _get_col(df, "game_state", "Game State")
    slot_lv_col = _get_col(df, "slot_lv", "slot level")
    slot_type_col = _get_col(df, "slot_type", "slotType", "slot type")
    agency_lv_col = _get_col(df, "agency_lv", "agencyLv", "agency level")

    if ev_col is None or time_col is None:
        raise ValueError("User CSV must include at least: Event Name, Time")

    keep = [
        c for c in [
            ev_col, time_col, did_col, stage_lv_col, stage_id_col, combat_col,
            game_state_col, slot_lv_col, slot_type_col, agency_lv_col
        ] if c is not None
    ]
    df = df[keep].copy()

    df.rename(columns={
        ev_col: "event_name",
        time_col: "time",
        (did_col or ""): "distinct_id",
        (stage_lv_col or ""): "stage_lv",
        (stage_id_col or ""): "stage_id",
        (combat_col or ""): "combat_power",
        (game_state_col or ""): "game_state",
        (slot_lv_col or ""): "slot_lv",
        (slot_type_col or ""): "slot_type",
        (agency_lv_col or ""): "agency_lv",
    }, inplace=True)

    for c in ["distinct_id", "stage_lv", "stage_id", "combat_power", "game_state", "slot_lv", "slot_type", "agency_lv"]:
        if c not in df.columns:
            df[c] = np.nan

    df["time"] = df["time"].apply(_to_float)
    df["event_name"] = df["event_name"].astype(str)

    df["stage_lv"] = df["stage_lv"].apply(_to_int)
    df["combat_power"] = df["combat_power"].apply(_to_float)

    df["slot_lv"] = df["slot_lv"].apply(_to_int)
    df["slot_type"] = df["slot_type"].apply(_to_int)
    df["agency_lv"] = df["agency_lv"].apply(_to_int)

    df["_gs"] = df["game_state"].apply(_safe_json_loads)

    return df.sort_values("time").reset_index(drop=True)


def slot_levels_to_columns(slot_lv_by_type_id: Dict[int, int]) -> Dict[str, int]:
    out = {}
    for i in range(1, 7):
        out[f"slot_lv_{i}"] = int(slot_lv_by_type_id.get(i, 0))
    return out


# =========================
# Build snapshots
# =========================
def build_stage_rows_for_user(
    df: pd.DataFrame,
    user_label: str,
    agency_map: Dict[int, AgencyRow],
    slotType_to_type_id: Dict[str, int],
    item_map: Dict[str, ItemRow],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    slot_lv_by_type_id: Dict[int, int] = {}
    agency_lv_current: int = 0

    rows = []
    validation = {
        "user": user_label,
        "unknown_item_ids_in_equips": set(),
        "unknown_item_ids_in_runes": set(),
        "missing_slotType_mapping": set(),
        "game_end_rows": 0,
        "first_clear_rows": 0,
        "parsed_game_state_fail": 0,
        "stage_clear_rows_emitted": 0,
    }

    for _, r in df.iterrows():
        ev = str(r["event_name"])

        if ev == "inventory_slot_lvup":
            stype = int(r.get("slot_type", 0) or 0)
            slv = int(r.get("slot_lv", 0) or 0)
            if stype > 0:
                slot_lv_by_type_id[stype] = slv

        if ev == "agency_lvup":
            alv = int(r.get("agency_lv", 0) or 0)
            if alv > 0:
                agency_lv_current = alv

        if ev == "game_end":
            validation["game_end_rows"] += 1
            gs = r.get("_gs")
            if not isinstance(gs, dict):
                validation["parsed_game_state_fail"] += 1
                continue

            first_clear = gs.get("first_clear", False)
            if isinstance(first_clear, str):
                first_clear = first_clear.lower() == "true"
            if not first_clear:
                continue
            validation["first_clear_rows"] += 1

            equips = gs.get("equips", []) or []
            runes = gs.get("runes", []) or []
            equips = [str(x) for x in equips]
            runes = [str(x) for x in runes]

            # Validate item ids + slotType mapping
            for eid in equips:
                it = item_map.get(eid)
                if not it:
                    validation["unknown_item_ids_in_equips"].add(eid)
                else:
                    if it.slotType not in slotType_to_type_id:
                        validation["missing_slotType_mapping"].add(it.slotType)

            for rid in runes:
                it = item_map.get(rid)
                if not it:
                    validation["unknown_item_ids_in_runes"].add(rid)

            stage_lv = int(r.get("stage_lv", 0) or 0)
            total_power = float(r.get("combat_power", 0.0) or 0.0)

            # agency atk/hp/power
            arow = agency_map.get(int(agency_lv_current))
            agencyAtk = float(arow.agencyAtk) if arow else 0.0
            agencyHp = float(arow.agencyHp) if arow else 0.0
            agency_power = (agencyAtk * 4.0 + agencyHp)

            (
                equip_power,
                rune_power,
                equip_base_power,
                equip_level_power,
                optionAtk_power_total,
                optionHp_power_total,
                optionAtk_power_self,
                optionHp_power_self,
                equip_atk_sum,
                equip_hp_sum,
                rune_atk_sum,
                rune_hp_sum,
                equip_names,
                rune_names,
                equip_grades,
                rune_grades,
            ) = sum_equip_rune_power(
                equip_ids=equips,
                rune_ids=runes,
                slot_lv_by_type_id=slot_lv_by_type_id,
                slotType_to_type_id=slotType_to_type_id,
                item_map=item_map,
                agency_atk=agencyAtk,
                agency_hp=agencyHp,
            )

            row = {
                "user": user_label,
                "time": float(r.get("time", 0.0) or 0.0),
                "stage_lv": stage_lv,
                "stage_id": str(r.get("stage_id", "")),
                "total_power": total_power,

                "equip_power": equip_power,
                "rune_power": rune_power,

                "agency_lv": int(agency_lv_current),
                "agencyAtk": agencyAtk,
                "agencyHp": agencyHp,
                "agency_power": agency_power,

                "equip_base_power": equip_base_power,
                "equip_level_power": equip_level_power,

                "equip_atk_sum": equip_atk_sum,
                "equip_hp_sum": equip_hp_sum,
                "rune_atk_sum": rune_atk_sum,
                "rune_hp_sum": rune_hp_sum,

                "equips_ids": equips,
                "runes_ids": runes,
                "equips_names": equip_names,
                "runes_names": rune_names,
                "equips_grades": equip_grades,
                "runes_grades": rune_grades,

                "optionAtk_power_total": optionAtk_power_total,
                "optionHp_power_total": optionHp_power_total,
                "optionAtk_power_self": optionAtk_power_self,
                "optionHp_power_self": optionHp_power_self,
            }
            row.update(slot_levels_to_columns(slot_lv_by_type_id))
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out, validation

    out = out.sort_values(["stage_lv", "time"]).drop_duplicates(subset=["user", "stage_lv"], keep="first")
    out = out.reset_index(drop=True)

    validation["stage_clear_rows_emitted"] = len(out)
    for k in ["unknown_item_ids_in_equips", "unknown_item_ids_in_runes", "missing_slotType_mapping"]:
        validation[k] = sorted(list(validation[k]))
    return out, validation


def build_breakdown_tables_for_snapshot(
    snapshot_row: pd.Series,
    slotType_to_type_id: Dict[str, int],
    item_map: Dict[str, ItemRow],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slot_levels = []
    slot_lv_by_type_id = {}
    for i in range(1, 7):
        lv = int(snapshot_row.get(f"slot_lv_{i}", 0) or 0)
        slot_lv_by_type_id[i] = lv
        slot_levels.append({"slot_type_id": i, "slot_lv": lv})
    slot_levels_df = pd.DataFrame(slot_levels)

    equips_ids = snapshot_row.get("equips_ids", [])
    runes_ids = snapshot_row.get("runes_ids", [])

    equip_rows = []
    for eid in equips_ids:
        it = item_map.get(str(eid))
        if not it:
            equip_rows.append({
                "name": "(lookup missing)",
                "equip_id": str(eid),
                "gradeType": "",
                "slotType": "",
                "slot_type_id": None,
                "slot_lv": None,
                "atk": 0.0,
                "hp": 0.0,
                "power": 0.0,
                "base_power(slot_lv=0)": 0.0,
                "level_power": 0.0,
                "optionAtk(ratio)": 0.0,
                "optionHp(ratio)": 0.0,
                "optionAtkBase(ratio)": 0.0,
                "optionHpBase(ratio)": 0.0,
            })
            continue

        type_id = slotType_to_type_id.get(it.slotType)
        lv = slot_lv_by_type_id.get(int(type_id), 0) if type_id is not None else 0

        atk, hp, p = calc_item_power(it, lv, is_equip=True)
        _, _, base = calc_item_power(it, 0, is_equip=True)

        equip_rows.append({
            "name": it.name,
            "equip_id": it.id,  # 검증용 유지
            "gradeType": it.gradeType,
            "slotType": it.slotType,
            "slot_type_id": type_id,
            "slot_lv": lv,
            "atk": atk,
            "hp": hp,
            "power": p,
            "base_power(slot_lv=0)": base,
            "level_power": p - base,
            "optionAtk(ratio)": it.optionAtk,
            "optionHp(ratio)": it.optionHp,
            "optionAtkBase(ratio)": it.optionAtkBase,
            "optionHpBase(ratio)": it.optionHpBase,
        })

    equip_df = pd.DataFrame(equip_rows)
    if not equip_df.empty:
        equip_df = equip_df.sort_values(["slot_type_id", "equip_id"], na_position="last")

    rune_rows = []
    for rid in runes_ids:
        it = item_map.get(str(rid))
        if not it:
            rune_rows.append({
                "name": "(lookup missing)",
                "rune_id": str(rid),
                "gradeType": "",
                "slotType": "",
                "atk": 0.0,
                "hp": 0.0,
                "power": 0.0,
                "optionAtk(ratio)": 0.0,
                "optionHp(ratio)": 0.0,
            })
            continue

        atk, hp, p = calc_item_power(it, 0, is_equip=False)
        rune_rows.append({
            "name": it.name,
            "rune_id": it.id,  # 검증용 유지
            "gradeType": it.gradeType,
            "slotType": it.slotType,
            "atk": atk,
            "hp": hp,
            "power": p,
            "optionAtk(ratio)": it.optionAtk,
            "optionHp(ratio)": it.optionHp,
        })

    rune_df = pd.DataFrame(rune_rows)
    return slot_levels_df, equip_df, rune_df


def names_join(xs: Any) -> str:
    if isinstance(xs, list):
        return ", ".join([str(x) for x in xs])
    if xs is None or (isinstance(xs, float) and np.isnan(xs)):
        return ""
    return str(xs)


# =========================
# Sidebar: Inputs
# =========================
with st.sidebar:
    st.header("입력")

    max_stage = st.number_input("최대 스테이지", min_value=1, max_value=2000, value=240, step=10)

    user_files = st.file_uploader(
        "유저 CSV 업로드 (1개~N개)",
        type=["csv"],
        accept_multiple_files=True,
    )

    agency_lookup_file = st.file_uploader(
        "룩업 CSV 1 (Agency) — 컬럼: agencyLv, agencyAtk, agencyHp",
        type=["csv"],
        accept_multiple_files=False,
    )

    item_lookup_file = st.file_uploader(
        "룩업 CSV 2 (Items: Equip+Rune) — 컬럼: id, name, slotType, atkBase, atkInc, hpBase, hpInc, option..., gradeType",
        type=["csv"],
        accept_multiple_files=False,
    )

    st.divider()
    st.subheader("slotType 매핑")
    st.caption("Item.slotType 문자열 → 유저 CSV의 slot_type(1~6)로 매핑")

    default_slotType_mapping = {
        "Hat": 1,
        "Coat": 2,
        "Ring": 3,
        "Neck": 4,
        "Belt": 5,
        "Boots": 6,
    }
    default_lines = "\n".join([f"{k}={v}" for k, v in default_slotType_mapping.items()])
    mapping_txt = st.text_area("매핑 편집", value=default_lines, height=160)

    slotType_to_type_id: Dict[str, int] = {}
    for line in mapping_txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        try:
            slotType_to_type_id[k] = int(v.strip())
        except Exception:
            continue
    if not slotType_to_type_id:
        slotType_to_type_id = default_slotType_mapping


# =========================
# Validate required uploads
# =========================
if not user_files:
    st.info("왼쪽에서 유저 CSV(1개 이상)를 업로드하세요.")
    st.stop()
if agency_lookup_file is None:
    st.info("왼쪽에서 Agency 룩업 CSV(agencyLv, agencyAtk, agencyHp)를 업로드하세요.")
    st.stop()
if item_lookup_file is None:
    st.info("왼쪽에서 Item 룩업 CSV(id, name, slotType, ... option..., gradeType)를 업로드하세요.")
    st.stop()

try:
    agency_map = load_agency_lookup(agency_lookup_file)
    item_map = load_item_lookup(item_lookup_file)
except Exception as e:
    st.error(f"룩업 파일 로딩 실패: {e}")
    st.stop()


# =========================
# Build per-user snapshots
# =========================
all_rows = []
validations = []
errors = []

for f in user_files:
    user_label = f.name.replace(".csv", "")
    try:
        udf = parse_user_csv(f)
        stage_rows, vinfo = build_stage_rows_for_user(
            df=udf,
            user_label=user_label,
            agency_map=agency_map,
            slotType_to_type_id=slotType_to_type_id,
            item_map=item_map,
        )
        validations.append(vinfo)
        if not stage_rows.empty:
            all_rows.append(stage_rows)
        else:
            errors.append(f"{user_label}: first_clear game_end 스냅샷이 없습니다.")
    except Exception as e:
        errors.append(f"{user_label}: 처리 실패 ({e})")

if not all_rows:
    st.error("유효한 stage clear 데이터가 없습니다. (조건: game_end + game_state.first_clear == true)")
    if errors:
        with st.expander("처리 오류", expanded=True):
            for msg in errors:
                st.write(f"- {msg}")
    st.stop()

df_all = pd.concat(all_rows, ignore_index=True)

# Stage range filter
df_all = df_all[(df_all["stage_lv"] >= 1) & (df_all["stage_lv"] <= int(max_stage))].copy()

# Derived sums + gap label
df_all["calc_sum_power"] = (
    df_all["equip_power"]
    + df_all["rune_power"]
    + df_all["agency_power"]
    + df_all["optionAtk_power_total"]
    + df_all["optionHp_power_total"]
    + df_all["optionAtk_power_self"]
    + df_all["optionHp_power_self"]
)
df_all["gap_total_minus_calc"] = df_all["total_power"] - df_all["calc_sum_power"]


def _gap_label(g: float) -> str:
    gi = int(round(float(g)))
    if gi == 0:
        return "정확"
    if gi > 0:
        return f"부족 {gi}"
    return f"초과 {abs(gi)}"


df_all["gap_label"] = df_all["gap_total_minus_calc"].apply(_gap_label)

# Name-based list strings for tables
df_all["equips"] = df_all["equips_names"].apply(names_join)
df_all["runes"] = df_all["runes_names"].apply(names_join)

# Grades list strings (table/debug)
df_all["equips_gradeTypes"] = df_all["equips_grades"].apply(names_join)
df_all["runes_gradeTypes"] = df_all["runes_grades"].apply(names_join)

# Keep ID strings too (for debugging)
df_all["equips_ids_str"] = df_all["equips_ids"].apply(names_join)
df_all["runes_ids_str"] = df_all["runes_ids"].apply(names_join)


# =========================
# UI: Tabs
# =========================
tab_overall, tab_user, tab_avg, tab_validate = st.tabs(["전체(그래프)", "유저별(그래프+검증)", "스테이지 평균", "데이터 오류 검증"])


# -------------------------
# Tab: Overall graphs + full table
# -------------------------
with tab_overall:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        stage_min = st.number_input("stage min", min_value=1, max_value=int(max_stage), value=1, step=1, key="overall_stage_min")
    with c2:
        stage_max = st.number_input("stage max", min_value=1, max_value=int(max_stage), value=int(max_stage), step=1, key="overall_stage_max")
    with c3:
        metric = st.selectbox(
            "지표",
            options=[
                "total_power", "equip_power", "rune_power", "agency_power",
                "agencyAtk", "agencyHp",
                "equip_base_power", "equip_level_power",
                "calc_sum_power", "gap_total_minus_calc",
                "optionAtk_power_total", "optionHp_power_total", "optionAtk_power_self", "optionHp_power_self",
                "equip_atk_sum", "equip_hp_sum", "rune_atk_sum", "rune_hp_sum",
            ],
            index=0,
            key="overall_metric",
        )
    with c4:
        users_sel = st.multiselect("유저(선택 시 필터)", options=sorted(df_all["user"].unique()), default=[], key="overall_users")

    view = df_all[(df_all["stage_lv"] >= int(stage_min)) & (df_all["stage_lv"] <= int(stage_max))].copy()
    if users_sel:
        view = view[view["user"].isin(users_sel)].copy()

    fig = px.line(
        view.sort_values(["user", "stage_lv"]),
        x="stage_lv",
        y=metric,
        color="user",
        markers=False,
        title=f"Stage vs {metric}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("전체 테이블 (유저×스테이지 스냅샷)")
    with st.expander("표 칼럼 표시 설정", expanded=False):
        all_cols = list(view.columns)
        selected_cols = st.multiselect("표에 표시할 칼럼", options=all_cols, default=all_cols, key="overall_table_cols")

    st.dataframe(view[selected_cols], use_container_width=True, height=520)


# -------------------------
# Tab: Per-user graphs + deep breakdown
# -------------------------
with tab_user:
    users = sorted(df_all["user"].unique())
    user_sel = st.selectbox("유저 선택", options=users, index=0, key="user_sel")

    udf = df_all[df_all["user"] == user_sel].sort_values("stage_lv").copy()

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        metric_u = st.selectbox(
            "지표",
            options=[
                "total_power", "equip_power", "rune_power", "agency_power",
                "agencyAtk", "agencyHp",
                "equip_base_power", "equip_level_power",
                "calc_sum_power", "gap_total_minus_calc",
                "optionAtk_power_total", "optionHp_power_total", "optionAtk_power_self", "optionHp_power_self",
                "equip_atk_sum", "equip_hp_sum", "rune_atk_sum", "rune_hp_sum",
            ],
            index=0,
            key="user_metric",
        )
    with c2:
        show_points = st.checkbox("포인트 표시", value=True, key="user_points")
    with c3:
        stage_focus = st.selectbox(
            "상세 확인할 스테이지(선택)",
            options=udf["stage_lv"].tolist(),
            index=len(udf) - 1 if len(udf) > 0 else 0,
            key="user_stage_focus",
        )

    fig2 = px.line(
        udf,
        x="stage_lv",
        y=metric_u,
        markers=bool(show_points),
        title=f"{user_sel}: Stage vs {metric_u}",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("선택 스테이지 상세 분해 (장비/룬/슬롯레벨 확인)")

    snap = udf[udf["stage_lv"] == int(stage_focus)].iloc[0]
    slot_levels_df, equip_df, rune_df = build_breakdown_tables_for_snapshot(
        snapshot_row=snap,
        slotType_to_type_id=slotType_to_type_id,
        item_map=item_map,
    )

    # Inline metrics (validation-friendly)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("total_power(로그)", f"{snap['total_power']:.0f}")
    s2.metric("equip_power(계산)", f"{snap['equip_power']:.0f}")
    s3.metric("rune_power(계산)", f"{snap['rune_power']:.0f}")
    s4.metric("agency_power(계산)", f"{snap['agency_power']:.0f}")

    s5, s6, s7, s8 = st.columns(4)
    s5.metric("calc_sum_power", f"{snap['calc_sum_power']:.0f}")
    s6.metric("gap_total_minus_calc", f"{snap['gap_total_minus_calc']:.0f}")
    s7.metric("gap_label", snap["gap_label"])
    s8.metric("agency_lv", str(int(snap["agency_lv"])))

    st.caption("gap_total_minus_calc = total_power - calc_sum_power. 0이면 '정확'.")

    cA, cB = st.columns([1, 2], gap="large")
    with cA:
        st.markdown("**슬롯 타입별 slot_lv (이 스테이지 클리어 시점 기준)**")
        st.dataframe(slot_levels_df, use_container_width=True, height=260)

    with cB:
        st.markdown("**장비 상세 (name 기준, id는 검증용)**")
        st.dataframe(equip_df, use_container_width=True, height=260)

    st.markdown("**룬 상세 (name 기준, id는 검증용)**")
    st.dataframe(rune_df, use_container_width=True, height=240)

    st.divider()
    st.subheader("유저 테이블")
    with st.expander("표 칼럼 표시 설정", expanded=False):
        all_cols_u = list(udf.columns)
        selected_cols_u = st.multiselect("표에 표시할 칼럼", options=all_cols_u, default=all_cols_u, key="user_table_cols")
    st.dataframe(udf[selected_cols_u], use_container_width=True, height=420)


# -------------------------
# Tab: Stage average
# -------------------------
with tab_avg:
    agg = (
        df_all
        .groupby("stage_lv", as_index=False)
        .agg(
            n_users=("user", "nunique"),
            avg_total_power=("total_power", "mean"),
            avg_equip_power=("equip_power", "mean"),
            avg_rune_power=("rune_power", "mean"),
            avg_agency_power=("agency_power", "mean"),
            avg_calc_sum_power=("calc_sum_power", "mean"),
            avg_gap=("gap_total_minus_calc", "mean"),
            top_gear_grade=("equips_grades", _mode_from_list_series),
            top_rune_grade=("runes_grades", _mode_from_list_series),
        )
        .sort_values("stage_lv")
    )

    c1, c2 = st.columns([1, 3])
    with c1:
        metric_a = st.selectbox(
            "평균 지표",
            options=[
                "avg_total_power", "avg_equip_power", "avg_rune_power", "avg_agency_power",
                "avg_calc_sum_power", "avg_gap"
            ],
            index=0,
            key="avg_metric",
        )
    with c2:
        fig3 = px.line(
            agg,
            x="stage_lv",
            y=metric_a,
            title=f"Stage vs {metric_a}",
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(agg, use_container_width=True, height=520)


# -------------------------
# Tab: Validation
# -------------------------
with tab_validate:
    st.subheader("데이터 오류 검증")

    if errors:
        with st.expander("파일 처리 오류", expanded=True):
            for msg in errors:
                st.write(f"- {msg}")

    vdf = pd.DataFrame(validations)
    if not vdf.empty:
        show_cols = [
            "user",
            "game_end_rows", "first_clear_rows", "parsed_game_state_fail", "stage_clear_rows_emitted",
            "unknown_item_ids_in_equips", "unknown_item_ids_in_runes", "missing_slotType_mapping",
        ]
        show_cols = [c for c in show_cols if c in vdf.columns]
        st.dataframe(vdf[show_cols], use_container_width=True, height=360)

    st.markdown("**전역 체크**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"- 업로드 유저 수: {df_all['user'].nunique()}")
        st.write(f"- 스냅샷 레코드 수: {len(df_all)}")
    with c2:
        unknown_e = sorted({x for v in validations for x in (v.get("unknown_item_ids_in_equips") or [])})
        unknown_r = sorted({x for v in validations for x in (v.get("unknown_item_ids_in_runes") or [])})
        if unknown_e:
            st.warning(f"장비 룩업 누락 ID 수: {len(unknown_e)}")
        else:
            st.success("장비 룩업 누락 ID: 없음")
        if unknown_r:
            st.warning(f"룬 룩업 누락 ID 수: {len(unknown_r)}")
        else:
            st.success("룬 룩업 누락 ID: 없음")
    with c3:
        missing_map = sorted({x for v in validations for x in (v.get("missing_slotType_mapping") or [])})
        if missing_map:
            st.warning(f"slotType 매핑 누락: {missing_map}")
        else:
            st.success("slotType 매핑 누락: 없음")

    with st.expander("룩업 누락/매핑 누락 상세", expanded=False):
        if unknown_e:
            st.markdown("**장비 룩업 누락 ID**")
            st.write(", ".join(unknown_e))
        if unknown_r:
            st.markdown("**룬 룩업 누락 ID**")
            st.write(", ".join(unknown_r))
        if missing_map:
            st.markdown("**slotType 매핑 누락(아이템 slotType 문자열)**")
            st.write(", ".join(missing_map))
