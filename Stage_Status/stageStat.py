import json
import ast
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


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


def _iround(x: float) -> int:
    # 소수점이 싫다 -> 최종 스탯/전투력 산출 시 정수 반올림
    return int(round(float(x)))


def _safe_div(n: float, d: float) -> float:
    return (n / d) if d != 0 else 0.0


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
    optionAtk: int          # 총합 기준 ATK% (정수: 5 => 5%)
    optionHp: int           # 총합 기준 HP%  (정수)
    optionAtkBase: int      # 자기 자신 기준 ATK% (정수)
    optionHpBase: int       # 자기 자신 기준 HP%  (정수)


def load_item_lookup(item_file) -> Dict[str, ItemRow]:
    df = pd.read_csv(item_file)
    df = _clean_cols(df)

    id_col = _get_col(df, "id")
    name_col = _get_col(df, "name")
    slot_col = _get_col(df, "slotType", "slottype", "slot_type")
    if id_col is None or name_col is None or slot_col is None:
        raise ValueError("Item lookup CSV must include: id, name, slotType, atkBase, atkInc, hpBase, hpInc (+ option columns optional)")

    for c in ["atkBase", "atkInc", "hpBase", "hpInc"]:
        cc = _get_col(df, c)
        if cc is None:
            df[c] = 0
        else:
            df[c] = pd.to_numeric(df[cc], errors="coerce").fillna(0)

    # 옵션 컬럼(없으면 0)
    for c in ["optionAtk", "optionHp", "optionAtkBase", "optionHpBase"]:
        cc = _get_col(df, c)
        if cc is None:
            df[c] = 0
        else:
            df[c] = pd.to_numeric(df[cc], errors="coerce").fillna(0).astype(int)

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
            optionAtk=int(r["optionAtk"]),
            optionHp=int(r["optionHp"]),
            optionAtkBase=int(r["optionAtkBase"]),
            optionHpBase=int(r["optionHpBase"]),
        )
    return out


def load_agency_lookup(agency_file) -> Dict[int, float]:
    df = pd.read_csv(agency_file)
    df = _clean_cols(df)

    lvl_col = _get_col(df, "lvl", "level", "agency_lv", "agency_level")
    pwr_col = _get_col(df, "agency_power", "power")
    if lvl_col is None or pwr_col is None:
        raise ValueError("Agency lookup CSV must include columns: lvl, agency_power")

    df["lvl_i"] = df[lvl_col].apply(_to_int)
    df["pwr_f"] = df[pwr_col].apply(_to_float)
    return dict(zip(df["lvl_i"], df["pwr_f"]))


# =========================
# Settings (user-tunable)
# =========================
def read_calc_settings_from_sidebar() -> Dict[str, Any]:
    with st.sidebar.expander("전투력 계산 설정", expanded=False):
        st.caption("기본값은 현재 기획(대화에서 확정된 디폴트)입니다. 필요 시 사용자가 수정 가능합니다.")

        st.subheader("전투력 환산 계수")
        atk_w = st.number_input("ATK 가중치", min_value=0.0, max_value=100.0, value=4.0, step=0.5)
        hp_w = st.number_input("HP 가중치", min_value=0.0, max_value=100.0, value=1.0, step=0.5)

        st.subheader("Agency 전투력 → ATK/HP 환산")
        st.caption("옵션 토탈(basis) 계산을 위해 Agency ATK/HP가 필요합니다.")
        agency_mode = st.selectbox(
            "Agency 전투력 합산 방식",
            options=["power_direct", "convert_to_stats"],
            index=0,
            help="power_direct: agency_power(룩업)를 그대로 합산 / convert_to_stats: ATK/HP로 환산 후 power로 재계산해 합산",
        )
        agency_atk_from_power = st.number_input("agency_atk = agency_power ×", min_value=0.0, max_value=10.0, value=0.25, step=0.01)
        agency_hp_from_power = st.number_input("agency_hp  = agency_power ×", min_value=0.0, max_value=10.0, value=0.50, step=0.01)

        st.subheader("장비 atkInc/hpInc 적용 방식")
        st.caption("권장 방식(안전한 모드): linear/step/pow/none")
        atk_mode = st.selectbox("atkInc 적용", options=["linear", "step", "pow", "none"], index=0)
        hp_mode = st.selectbox("hpInc 적용", options=["linear", "step", "pow", "none"], index=0)
        atk_step_k = st.number_input("atk step k(단계형일 때)", min_value=1, max_value=999, value=5, step=1)
        hp_step_k = st.number_input("hp  step k(단계형일 때)", min_value=1, max_value=999, value=5, step=1)
        atk_pow_p = st.number_input("atk pow p(거듭제곱일 때)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        hp_pow_p = st.number_input("hp  pow p(거듭제곱일 때)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

        st.subheader("옵션 토탈(optionAtk/optionHp) – % 적용 기준(basis)")
        basis_equip = st.checkbox("basis에 장비(최종 ATK/HP) 포함", value=True)
        basis_rune = st.checkbox("basis에 룬 ATK/HP 포함", value=True)
        basis_agency = st.checkbox("basis에 에이전시 ATK/HP 포함", value=True)
        basis_equip_level = st.checkbox("basis에 장비 레벨 ATK/HP 포함", value=True)
        pct_mode = st.selectbox("총합 기준 % 합산 방식", options=["sum", "compound"], index=0, help="sum: % 단순합 / compound: (1+p1)(1+p2)...-1")

        st.subheader("옵션 셀프(optionAtkBase/optionHpBase) – % 적용 기준")
        self_basis = st.selectbox("장비 자기 자신 기준", options=["self_final_stats", "self_base_stats"], index=0)

    return {
        "POWER_ATK_WEIGHT": float(atk_w),
        "POWER_HP_WEIGHT": float(hp_w),
        "AGENCY_MODE": agency_mode,
        "AGENCY_ATK_FROM_POWER": float(agency_atk_from_power),
        "AGENCY_HP_FROM_POWER": float(agency_hp_from_power),
        "EQUIP_ATK_INC_MODE": atk_mode,
        "EQUIP_HP_INC_MODE": hp_mode,
        "EQUIP_ATK_STEP_K": int(atk_step_k),
        "EQUIP_HP_STEP_K": int(hp_step_k),
        "EQUIP_ATK_POW_P": float(atk_pow_p),
        "EQUIP_HP_POW_P": float(hp_pow_p),
        "OPTION_TOTAL_BASIS_EQUIP": bool(basis_equip),
        "OPTION_TOTAL_BASIS_RUNE": bool(basis_rune),
        "OPTION_TOTAL_BASIS_AGENCY": bool(basis_agency),
        "OPTION_TOTAL_BASIS_EQUIP_LEVEL": bool(basis_equip_level),
        "OPTION_TOTAL_PCT_MODE": pct_mode,
        "OPTION_SELF_BASIS": self_basis,
    }


def power_from_stats(atk: float, hp: float, settings: Dict[str, Any]) -> float:
    return atk * settings["POWER_ATK_WEIGHT"] + hp * settings["POWER_HP_WEIGHT"]


def apply_inc_mode(base: float, inc: float, lv: int, mode: str, step_k: int, pow_p: float) -> float:
    if mode == "none":
        return base
    if mode == "linear":
        return base + inc * lv
    if mode == "step":
        return base + inc * (lv // max(step_k, 1))
    if mode == "pow":
        return base + inc * (lv ** pow_p)
    return base + inc * lv


def calc_equip_stats(item: ItemRow, slot_lv: int, settings: Dict[str, Any]) -> Tuple[float, float]:
    atk = apply_inc_mode(item.atkBase, item.atkInc, slot_lv,
                         settings["EQUIP_ATK_INC_MODE"], settings["EQUIP_ATK_STEP_K"], settings["EQUIP_ATK_POW_P"])
    hp = apply_inc_mode(item.hpBase, item.hpInc, slot_lv,
                        settings["EQUIP_HP_INC_MODE"], settings["EQUIP_HP_STEP_K"], settings["EQUIP_HP_POW_P"])
    # 정수화(표/검증에서 소수점 방지)
    return float(_iround(atk)), float(_iround(hp))


def calc_rune_stats(item: ItemRow) -> Tuple[float, float]:
    # 룬은 Inc/slot_lv 적용 없음
    return float(_iround(item.atkBase)), float(_iround(item.hpBase))


def agency_stats_from_power(agency_power: float, settings: Dict[str, Any]) -> Tuple[float, float]:
    atk = agency_power * settings["AGENCY_ATK_FROM_POWER"]
    hp = agency_power * settings["AGENCY_HP_FROM_POWER"]
    return float(_iround(atk)), float(_iround(hp))


def combine_pct(pcts: List[float], mode: str) -> float:
    # pcts: 예) [0.05, 0.10]
    if not pcts:
        return 0.0
    if mode == "compound":
        mult = 1.0
        for p in pcts:
            mult *= (1.0 + p)
        return mult - 1.0
    return float(sum(pcts))


# =========================
# Core calculators
# =========================
def sum_equip_rune_agency_with_options(
    equip_ids: List[str],
    rune_ids: List[str],
    slot_lv_by_type_id: Dict[int, int],
    slotType_to_type_id: Dict[str, int],
    item_map: Dict[str, ItemRow],
    agency_power: float,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Returns a dict with:
      - equip_base_atk/hp, equip_level_atk/hp, equip_total_atk/hp
      - rune_atk/hp
      - agency_atk/hp
      - equip_base_power, equip_level_power, rune_power, agency_power_contrib
      - option_total_* , option_self_*
      - equip_power_for_chart (장비 전투력 = equip_base_power + option_total_power + option_self_power)
      - other_option_power (gap 대응용은 밖에서 계산)
      - equip_names/rune_names
    """
    equip_names: List[str] = []
    rune_names: List[str] = []

    # --- Equip stats split (base vs level) ---
    equip_base_atk = 0
    equip_base_hp = 0
    equip_total_atk = 0
    equip_total_hp = 0

    # option-self (장비 자기 자신 기준 %)
    option_self_atk_bonus = 0
    option_self_hp_bonus = 0

    # option-total (총합 기준 %) -> % 원천(아이템 옵션 합산)
    option_total_atk_pcts: List[float] = []
    option_total_hp_pcts: List[float] = []

    # iterate equips
    for eid in equip_ids:
        it = item_map.get(str(eid))
        if not it:
            equip_names.append(f"(missing:{eid})")
            continue
        equip_names.append(it.name)

        type_id = slotType_to_type_id.get(it.slotType)
        lv = slot_lv_by_type_id.get(int(type_id), 0) if type_id is not None else 0

        atk_final, hp_final = calc_equip_stats(it, lv, settings)
        atk_base, hp_base = calc_equip_stats(it, 0, settings)

        equip_total_atk += _iround(atk_final)
        equip_total_hp += _iround(hp_final)
        equip_base_atk += _iround(atk_base)
        equip_base_hp += _iround(hp_base)

        # optionAtk/optionHp: 총합 기준 % (장비도 포함)
        if it.optionAtk:
            option_total_atk_pcts.append(it.optionAtk / 100.0)
        if it.optionHp:
            option_total_hp_pcts.append(it.optionHp / 100.0)

        # optionAtkBase/optionHpBase: 자기 자신 기준 % (장비만)
        if settings["OPTION_SELF_BASIS"] == "self_base_stats":
            basis_atk_for_self = atk_base
            basis_hp_for_self = hp_base
        else:
            basis_atk_for_self = atk_final
            basis_hp_for_self = hp_final

        if it.optionAtkBase:
            option_self_atk_bonus += _iround(basis_atk_for_self * (it.optionAtkBase / 100.0))
        if it.optionHpBase:
            option_self_hp_bonus += _iround(basis_hp_for_self * (it.optionHpBase / 100.0))

    equip_level_atk = equip_total_atk - equip_base_atk
    equip_level_hp = equip_total_hp - equip_base_hp

    # --- Runes ---
    rune_atk = 0
    rune_hp = 0
    for rid in rune_ids:
        it = item_map.get(str(rid))
        if not it:
            rune_names.append(f"(missing:{rid})")
            continue
        rune_names.append(it.name)

        atk_r, hp_r = calc_rune_stats(it)
        rune_atk += _iround(atk_r)
        rune_hp += _iround(hp_r)

        # optionAtk/optionHp: 총합 기준 % (룬도 포함)
        if it.optionAtk:
            option_total_atk_pcts.append(it.optionAtk / 100.0)
        if it.optionHp:
            option_total_hp_pcts.append(it.optionHp / 100.0)

    # --- Agency stats (for option basis) ---
    agency_atk, agency_hp = agency_stats_from_power(agency_power, settings)

    # --- Build option-total basis (디폴트: equip/rune/agency/equip_level 포함) ---
    basis_atk = 0
    basis_hp = 0

    # 주의: "장비(최종)"과 "장비레벨"을 동시에 포함하면 중복이므로
    # 여기서는 요구사항대로 '장비/룬/에이전시/장비레벨'을 "컴포넌트 합산"으로 구성:
    #  - 장비(베이스) + 장비레벨 + 룬 + 에이전시
    if settings["OPTION_TOTAL_BASIS_EQUIP"]:
        basis_atk += equip_base_atk
        basis_hp += equip_base_hp
    if settings["OPTION_TOTAL_BASIS_EQUIP_LEVEL"]:
        basis_atk += equip_level_atk
        basis_hp += equip_level_hp
    if settings["OPTION_TOTAL_BASIS_RUNE"]:
        basis_atk += rune_atk
        basis_hp += rune_hp
    if settings["OPTION_TOTAL_BASIS_AGENCY"]:
        basis_atk += _iround(agency_atk)
        basis_hp += _iround(agency_hp)

    pct_atk = combine_pct(option_total_atk_pcts, settings["OPTION_TOTAL_PCT_MODE"])
    pct_hp = combine_pct(option_total_hp_pcts, settings["OPTION_TOTAL_PCT_MODE"])

    option_total_atk_bonus = _iround(basis_atk * pct_atk)
    option_total_hp_bonus = _iround(basis_hp * pct_hp)

    # --- Convert to power contributions ---
    equip_base_power = power_from_stats(equip_base_atk, equip_base_hp, settings)
    equip_level_power = power_from_stats(equip_level_atk, equip_level_hp, settings)
    rune_power = power_from_stats(rune_atk, rune_hp, settings)

    if settings["AGENCY_MODE"] == "convert_to_stats":
        agency_power_contrib = power_from_stats(agency_atk, agency_hp, settings)
    else:
        agency_power_contrib = float(agency_power)

    option_total_power = power_from_stats(option_total_atk_bonus, option_total_hp_bonus, settings)
    option_self_power = power_from_stats(option_self_atk_bonus, option_self_hp_bonus, settings)

    # 요구사항: "장비 전투력"에는 옵션 셀프 + 옵션 토탈 합쳐진 것 포함
    # + 장비 베이스 전투력(=slot_lv=0 기준)을 함께 장비 전투력으로 보기(레벨은 별도 시리즈로)
    equip_power_for_chart = equip_base_power + option_total_power + option_self_power

    return {
        "equip_names": equip_names,
        "rune_names": rune_names,

        "equip_base_atk": equip_base_atk,
        "equip_base_hp": equip_base_hp,
        "equip_level_atk": equip_level_atk,
        "equip_level_hp": equip_level_hp,
        "equip_total_atk": equip_total_atk,
        "equip_total_hp": equip_total_hp,

        "rune_atk": rune_atk,
        "rune_hp": rune_hp,

        "agency_power_lookup": float(agency_power),
        "agency_atk": _iround(agency_atk),
        "agency_hp": _iround(agency_hp),

        "basis_atk_for_option_total": int(basis_atk),
        "basis_hp_for_option_total": int(basis_hp),
        "option_total_pct_atk": float(pct_atk),
        "option_total_pct_hp": float(pct_hp),

        "option_total_atk_bonus": int(option_total_atk_bonus),
        "option_total_hp_bonus": int(option_total_hp_bonus),
        "option_self_atk_bonus": int(option_self_atk_bonus),
        "option_self_hp_bonus": int(option_self_hp_bonus),

        "equip_base_power": float(equip_base_power),
        "equip_level_power": float(equip_level_power),
        "rune_power": float(rune_power),
        "agency_power": float(agency_power_contrib),

        "option_total_power": float(option_total_power),
        "option_self_power": float(option_self_power),

        "equip_power_for_chart": float(equip_power_for_chart),
    }


# =========================
# Parsing user CSV
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
    agency_lv_col = _get_col(df, "agency_lv", "agency level")

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


def names_join(xs: Any) -> str:
    if isinstance(xs, list):
        return ", ".join([str(x) for x in xs])
    if xs is None or (isinstance(xs, float) and np.isnan(xs)):
        return ""
    return str(xs)


# =========================
# Build per-user stage rows
# =========================
def build_stage_rows_for_user(
    df: pd.DataFrame,
    user_label: str,
    agency_power_map: Dict[int, float],
    slotType_to_type_id: Dict[str, int],
    item_map: Dict[str, ItemRow],
    settings: Dict[str, Any],
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

            # validate item ids + slotType mapping
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

            agency_power_lookup = float(agency_power_map.get(int(agency_lv_current), 0.0))

            calc = sum_equip_rune_agency_with_options(
                equip_ids=equips,
                rune_ids=runes,
                slot_lv_by_type_id=slot_lv_by_type_id,
                slotType_to_type_id=slotType_to_type_id,
                item_map=item_map,
                agency_power=agency_power_lookup,
                settings=settings,
            )

            row = {
                "user": user_label,
                "time": float(r.get("time", 0.0) or 0.0),
                "stage_lv": stage_lv,
                "stage_id": str(r.get("stage_id", "")),
                "total_power": total_power,

                # chart components (요구된 이름 기반)
                "equip_power_chart": calc["equip_power_for_chart"],  # 장비 전투력(베이스+옵션)
                "rune_power": calc["rune_power"],                    # 룬 전투력
                "equip_level_power": calc["equip_level_power"],      # 장비 레벨 전투력
                "agency_power": calc["agency_power"],                # 에이전시 레벨 전투력

                # detailed breakdown (유지/검증용)
                "equip_base_power": calc["equip_base_power"],
                "option_total_power": calc["option_total_power"],
                "option_self_power": calc["option_self_power"],

                "agency_lv": int(agency_lv_current),
                "agency_power_lookup": calc["agency_power_lookup"],
                "agency_atk": calc["agency_atk"],
                "agency_hp": calc["agency_hp"],

                "equip_base_atk": calc["equip_base_atk"],
                "equip_base_hp": calc["equip_base_hp"],
                "equip_level_atk": calc["equip_level_atk"],
                "equip_level_hp": calc["equip_level_hp"],
                "equip_total_atk": calc["equip_total_atk"],
                "equip_total_hp": calc["equip_total_hp"],

                "rune_atk": calc["rune_atk"],
                "rune_hp": calc["rune_hp"],

                "basis_atk_for_option_total": calc["basis_atk_for_option_total"],
                "basis_hp_for_option_total": calc["basis_hp_for_option_total"],
                "option_total_pct_atk": calc["option_total_pct_atk"],
                "option_total_pct_hp": calc["option_total_pct_hp"],
                "option_total_atk_bonus": calc["option_total_atk_bonus"],
                "option_total_hp_bonus": calc["option_total_hp_bonus"],
                "option_self_atk_bonus": calc["option_self_atk_bonus"],
                "option_self_hp_bonus": calc["option_self_hp_bonus"],

                "equips_ids": equips,
                "runes_ids": runes,
                "equips_names": calc["equip_names"],
                "runes_names": calc["rune_names"],
            }
            row.update(slot_levels_to_columns(slot_lv_by_type_id))
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out, validation

    out = out.sort_values(["stage_lv", "time"]).drop_duplicates(subset=["user", "stage_lv"], keep="first").reset_index(drop=True)
    validation["stage_clear_rows_emitted"] = len(out)
    for k in ["unknown_item_ids_in_equips", "unknown_item_ids_in_runes", "missing_slotType_mapping"]:
        validation[k] = sorted(list(validation[k]))
    return out, validation


# =========================
# Breakdown tables (유저 탭 상세)
# =========================
def build_breakdown_tables_for_snapshot(
    snapshot_row: pd.Series,
    slotType_to_type_id: Dict[str, int],
    item_map: Dict[str, ItemRow],
    settings: Dict[str, Any],
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
                "slotType": "",
                "slot_type_id": None,
                "slot_lv": None,
                "atk_final": 0,
                "hp_final": 0,
                "atk_base(lv0)": 0,
                "hp_base(lv0)": 0,
                "base_power(lv0)": 0,
                "level_power": 0,
                "opt_total_%(atk)": 0,
                "opt_total_%(hp)": 0,
                "opt_self_%(atk)": 0,
                "opt_self_%(hp)": 0,
            })
            continue

        type_id = slotType_to_type_id.get(it.slotType)
        lv = slot_lv_by_type_id.get(int(type_id), 0) if type_id is not None else 0

        atk_final, hp_final = calc_equip_stats(it, lv, settings)
        atk_base, hp_base = calc_equip_stats(it, 0, settings)

        base_p = power_from_stats(atk_base, hp_base, settings)
        final_p = power_from_stats(atk_final, hp_final, settings)

        equip_rows.append({
            "name": it.name,
            "equip_id": it.id,  # 검증용 유지
            "slotType": it.slotType,
            "slot_type_id": type_id,
            "slot_lv": lv,
            "atk_final": int(_iround(atk_final)),
            "hp_final": int(_iround(hp_final)),
            "atk_base(lv0)": int(_iround(atk_base)),
            "hp_base(lv0)": int(_iround(hp_base)),
            "base_power(lv0)": int(_iround(base_p)),
            "level_power": int(_iround(final_p - base_p)),
            "opt_total_%(atk)": int(it.optionAtk),
            "opt_total_%(hp)": int(it.optionHp),
            "opt_self_%(atk)": int(it.optionAtkBase),
            "opt_self_%(hp)": int(it.optionHpBase),
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
                "slotType": "",
                "atk": 0,
                "hp": 0,
                "power": 0,
                "opt_total_%(atk)": 0,
                "opt_total_%(hp)": 0,
            })
            continue

        atk, hp = calc_rune_stats(it)
        p = power_from_stats(atk, hp, settings)

        rune_rows.append({
            "name": it.name,
            "rune_id": it.id,  # 검증용 유지
            "slotType": it.slotType,
            "atk": int(_iround(atk)),
            "hp": int(_iround(hp)),
            "power": int(_iround(p)),
            "opt_total_%(atk)": int(it.optionAtk),
            "opt_total_%(hp)": int(it.optionHp),
        })

    rune_df = pd.DataFrame(rune_rows)
    return slot_levels_df, equip_df, rune_df


# =========================
# Chart helper (one graph, many series, legend toggle)
# =========================
SERIES_LABELS = {
    "total_power": "전체 전투력",
    "equip_power_chart": "장비 전투력",
    "rune_power": "룬 전투력",
    "equip_level_power": "장비 레벨 전투력",
    "agency_power": "에이전시 레벨 전투력",
    "other_option_power": "그 외 옵션 전투력",
}

SERIES_KEYS = ["total_power", "equip_power_chart", "rune_power", "equip_level_power", "agency_power", "other_option_power"]


def build_multi_series_figure(df: pd.DataFrame, x_col: str, group_col: str, enabled: Dict[str, bool], title: str) -> go.Figure:
    fig = go.Figure()
    # 각 시리즈에 대해 user별 trace를 추가(legend는 클릭 토글)
    for key in SERIES_KEYS:
        if key not in df.columns:
            continue
        visible_default = True if enabled.get(key, True) else "legendonly"
        for g in sorted(df[group_col].unique()):
            sub = df[df[group_col] == g].sort_values(x_col)
            fig.add_trace(go.Scatter(
                x=sub[x_col],
                y=sub[key],
                mode="lines",
                name=f"{g} – {SERIES_LABELS[key]}",
                visible=visible_default,
            ))
    fig.update_layout(
        title=title,
        legend_title_text="범례 클릭으로 ON/OFF",
        hovermode="x unified",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(title=x_col)
    fig.update_yaxes(title="power")
    return fig


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
        "룩업 CSV 1 (Agency) — 컬럼: lvl, agency_power",
        type=["csv"],
        accept_multiple_files=False,
    )

    item_lookup_file = st.file_uploader(
        "룩업 CSV 2 (Items: Equip+Rune) — 컬럼: id, name, slotType, atkBase, atkInc, hpBase, hpInc (+ optionAtk/optionHp/optionAtkBase/optionHpBase)",
        type=["csv"],
        accept_multiple_files=False,
    )

    st.divider()
    st.subheader("slotType 매핑(편집 가능)")
    st.caption("Item.slotType 문자열 → 유저 CSV의 slot_type(1~6)로 매핑")

    # ✅ 요구 slotType 기본값 반영
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

# ✅ 계산 설정(사이드바 expander)
settings = read_calc_settings_from_sidebar()


# =========================
# Validate required uploads
# =========================
if not user_files:
    st.info("왼쪽에서 유저 CSV(1개 이상)를 업로드하세요.")
    st.stop()
if agency_lookup_file is None:
    st.info("왼쪽에서 Agency 룩업 CSV(lvl, agency_power)를 업로드하세요.")
    st.stop()
if item_lookup_file is None:
    st.info("왼쪽에서 Item 룩업 CSV(id, name, slotType, atkBase, atkInc, hpBase, hpInc ...)를 업로드하세요.")
    st.stop()

try:
    agency_power_map = load_agency_lookup(agency_lookup_file)
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
            agency_power_map=agency_power_map,
            slotType_to_type_id=slotType_to_type_id,
            item_map=item_map,
            settings=settings,
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

# Name-based list strings for tables
df_all["equips"] = df_all["equips_names"].apply(names_join)
df_all["runes"] = df_all["runes_names"].apply(names_join)
df_all["equips_ids_str"] = df_all["equips_ids"].apply(names_join)
df_all["runes_ids_str"] = df_all["runes_ids"].apply(names_join)

# 합산(계산 전투력) + gap
# - 장비 전투력은 chart용(베이스+옵션)이고, calc_sum에서는 "각 항목을 명시적으로" 합산
df_all["calc_sum_power"] = (
    df_all["equip_base_power"]
    + df_all["equip_level_power"]
    + df_all["rune_power"]
    + df_all["agency_power"]
    + df_all["option_total_power"]
    + df_all["option_self_power"]
)
df_all["gap_total_minus_calc"] = df_all["total_power"] - df_all["calc_sum_power"]
df_all["other_option_power"] = df_all["gap_total_minus_calc"]  # ✅ "그 외 옵션 전투력" = gap

def _gap_label(g: float) -> str:
    gi = _iround(g)
    if gi == 0:
        return "정확"
    if gi > 0:
        return f"부족 {gi}"
    return f"초과 {abs(gi)}"

df_all["gap_label"] = df_all["gap_total_minus_calc"].apply(_gap_label)


# =========================
# UI: Tabs
# =========================
tab_overall, tab_user, tab_avg, tab_validate = st.tabs(
    ["전체(그래프)", "유저별(그래프+검증)", "스테이지 평균", "데이터 오류 검증"]
)


# -------------------------
# Tab: Overall (one graph with toggles + full table)
# -------------------------
with tab_overall:
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        stage_min = st.number_input("stage min", min_value=1, max_value=int(max_stage), value=1, step=1, key="overall_stage_min")
    with c2:
        stage_max = st.number_input("stage max", min_value=1, max_value=int(max_stage), value=int(max_stage), step=1, key="overall_stage_max")
    with c3:
        users_sel = st.multiselect("유저(선택 시 필터)", options=sorted(df_all["user"].unique()), default=[], key="overall_users")

    view = df_all[(df_all["stage_lv"] >= int(stage_min)) & (df_all["stage_lv"] <= int(stage_max))].copy()
    if users_sel:
        view = view[view["user"].isin(users_sel)].copy()

    with st.expander("그래프 표시 설정(시리즈 ON/OFF 기본값)", expanded=False):
        enabled = {}
        for k in SERIES_KEYS:
            enabled[k] = st.checkbox(SERIES_LABELS[k], value=True, key=f"overall_series_{k}")
        st.caption("그래프에서 범례(legend)를 클릭하면 시리즈를 껐다 켰다 할 수 있어요.")

    fig = build_multi_series_figure(
        df=view,
        x_col="stage_lv",
        group_col="user",
        enabled=enabled,
        title="Stage vs (전체/장비/룬/장비레벨/에이전시/그외)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("전체 테이블 (유저×스테이지 스냅샷)")

    with st.expander("표 칼럼 표시 설정", expanded=False):
        all_cols = list(view.columns)
        selected_cols = st.multiselect("표에 표시할 칼럼", options=all_cols, default=all_cols, key="overall_table_cols")
    st.dataframe(view[selected_cols], use_container_width=True, height=520)


# -------------------------
# Tab: Per-user
# -------------------------
with tab_user:
    users = sorted(df_all["user"].unique())
    user_sel = st.selectbox("유저 선택", options=users, index=0, key="user_sel")

    udf = df_all[df_all["user"] == user_sel].sort_values("stage_lv").copy()

    c1, c2 = st.columns([2, 1])
    with c1:
        stage_focus = st.selectbox(
            "상세 확인할 스테이지(선택)",
            options=udf["stage_lv"].tolist(),
            index=len(udf) - 1 if len(udf) > 0 else 0,
            key="user_stage_focus",
        )
    with c2:
        show_points = st.checkbox("포인트 표시(참고)", value=False, key="user_points")  # 기존 UI 최대 유지

    with st.expander("그래프 표시 설정(시리즈 ON/OFF 기본값)", expanded=False):
        enabled_u = {}
        for k in SERIES_KEYS:
            enabled_u[k] = st.checkbox(SERIES_LABELS[k], value=True, key=f"user_series_{k}")
        st.caption("범례 클릭으로도 ON/OFF 가능합니다.")

    fig2 = build_multi_series_figure(
        df=udf,
        x_col="stage_lv",
        group_col="user",
        enabled=enabled_u,
        title=f"{user_sel}: Stage vs (전체/장비/룬/장비레벨/에이전시/그외)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("선택 스테이지 상세 분해 (장비/룬/슬롯레벨/옵션 확인)")

    snap = udf[udf["stage_lv"] == int(stage_focus)].iloc[0]
    slot_levels_df, equip_df, rune_df = build_breakdown_tables_for_snapshot(
        snapshot_row=snap,
        slotType_to_type_id=slotType_to_type_id,
        item_map=item_map,
        settings=settings,
    )

    # Inline metrics
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("전체 전투력(로그)", f"{snap['total_power']:.0f}")
    s2.metric("장비 전투력(=베이스+옵션)", f"{snap['equip_power_chart']:.0f}")
    s3.metric("룬 전투력", f"{snap['rune_power']:.0f}")
    s4.metric("장비 레벨 전투력", f"{snap['equip_level_power']:.0f}")
    s5.metric("에이전시 레벨 전투력", f"{snap['agency_power']:.0f}")
    s6.metric("그 외 옵션 전투력(gap)", f"{snap['other_option_power']:.0f}")

    s7, s8, s9, s10 = st.columns(4)
    s7.metric("calc_sum_power", f"{snap['calc_sum_power']:.0f}")
    s8.metric("gap_total_minus_calc", f"{snap['gap_total_minus_calc']:.0f}")
    s9.metric("gap_label", snap["gap_label"])
    s10.metric("agency_lv", str(int(snap["agency_lv"])))

    st.caption("그 외 옵션 전투력 = 전체 전투력 - (장비베이스 + 장비레벨 + 룬 + 에이전시 + 옵션토탈 + 옵션셀프)")

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
            total_power=("total_power", "mean"),
            equip_power_chart=("equip_power_chart", "mean"),
            rune_power=("rune_power", "mean"),
            equip_level_power=("equip_level_power", "mean"),
            agency_power=("agency_power", "mean"),
            other_option_power=("other_option_power", "mean"),
        )
        .sort_values("stage_lv")
    )

    with st.expander("그래프 표시 설정(시리즈 ON/OFF 기본값)", expanded=False):
        enabled_a = {}
        for k in SERIES_KEYS:
            enabled_a[k] = st.checkbox(SERIES_LABELS[k], value=True, key=f"avg_series_{k}")
        st.caption("범례 클릭으로도 ON/OFF 가능합니다.")

    fig3 = build_multi_series_figure(
        df=agg.assign(user="AVG"),  # group_col 맞추기 위해 임시 user 컬럼
        x_col="stage_lv",
        group_col="user",
        enabled=enabled_a,
        title="Stage 평균: (전체/장비/룬/장비레벨/에이전시/그외)",
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(agg, use_container_width=True, height=520)


# -------------------------
# Tab: Validation + Example calculator
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

    st.divider()
    st.subheader("예시 계산기 (설정값으로 전투력 재계산 확인)")

    # (A) 실제 스냅샷 선택 → 현재 설정으로 계산값을 다시 보여주기
    with st.expander("A) 스냅샷으로 예시 확인(권장)", expanded=True):
        u_sel = st.selectbox("유저", options=sorted(df_all["user"].unique()), index=0, key="ex_user")
        d_sel = df_all[df_all["user"] == u_sel].sort_values("stage_lv")
        st_sel = st.selectbox("stage_lv", options=d_sel["stage_lv"].tolist(), index=len(d_sel)-1, key="ex_stage")
        ex = d_sel[d_sel["stage_lv"] == int(st_sel)].iloc[0]

        # 현재 row에 이미 settings 적용된 결과가 들어있지만, "검산" 느낌으로 핵심만 다시 보여줌
        st.write("**현재 설정으로 산출된 값(스냅샷)**")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("전체 전투력", int(_iround(ex["total_power"])))
        m2.metric("장비 전투력", int(_iround(ex["equip_power_chart"])))
        m3.metric("룬 전투력", int(_iround(ex["rune_power"])))
        m4.metric("장비 레벨 전투력", int(_iround(ex["equip_level_power"])))
        m5.metric("에이전시 레벨 전투력", int(_iround(ex["agency_power"])))
        m6.metric("그 외 옵션 전투력", int(_iround(ex["other_option_power"])))

        st.caption("※ 계산 설정을 바꾸면(사이드바) 이 값들도 즉시 바뀝니다. → 수식/파라미터가 반영되었는지 확인할 때 사용하세요.")

    # (B) 입력 샌드박스: 임의의 ATK/HP를 넣어서 옵션 토탈/셀프를 테스트
    with st.expander("B) 숫자 샌드박스(간단 테스트)", expanded=False):
        st.caption("원하는 값 1개를 넣고, 현재 설정에서 옵션이 얼마나 붙는지 빠르게 확인합니다(정수 반올림).")
        b_equip_base_atk = st.number_input("장비 베이스 ATK", min_value=0, value=1000, step=10)
        b_equip_base_hp = st.number_input("장비 베이스 HP", min_value=0, value=2000, step=10)
        b_equip_lv_atk = st.number_input("장비 레벨 ATK(증가분)", min_value=0, value=300, step=10)
        b_equip_lv_hp = st.number_input("장비 레벨 HP(증가분)", min_value=0, value=600, step=10)
        b_rune_atk = st.number_input("룬 ATK", min_value=0, value=200, step=10)
        b_rune_hp = st.number_input("룬 HP", min_value=0, value=400, step=10)
        b_agency_power = st.number_input("에이전시 전투력(룩업값)", min_value=0, value=5000, step=100)

        b_opt_atk_pct = st.number_input("optionAtk 총합 %(예: 5)", min_value=0, value=5, step=1)
        b_opt_hp_pct = st.number_input("optionHp 총합 %(예: 10)", min_value=0, value=0, step=1)

        # basis 구성(설정값 반영)
        a_atk, a_hp = agency_stats_from_power(b_agency_power, settings)

        basis_atk = 0
        basis_hp = 0
        if settings["OPTION_TOTAL_BASIS_EQUIP"]:
            basis_atk += int(b_equip_base_atk)
            basis_hp += int(b_equip_base_hp)
        if settings["OPTION_TOTAL_BASIS_EQUIP_LEVEL"]:
            basis_atk += int(b_equip_lv_atk)
            basis_hp += int(b_equip_lv_hp)
        if settings["OPTION_TOTAL_BASIS_RUNE"]:
            basis_atk += int(b_rune_atk)
            basis_hp += int(b_rune_hp)
        if settings["OPTION_TOTAL_BASIS_AGENCY"]:
            basis_atk += int(a_atk)
            basis_hp += int(a_hp)

        # pct mode 반영(샌드박스는 단일 %로 단순)
        pct_atk = (b_opt_atk_pct / 100.0)
        pct_hp = (b_opt_hp_pct / 100.0)

        opt_total_atk_bonus = _iround(basis_atk * pct_atk)
        opt_total_hp_bonus = _iround(basis_hp * pct_hp)
        opt_total_power = power_from_stats(opt_total_atk_bonus, opt_total_hp_bonus, settings)

        st.write("**결과**")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("basis ATK", basis_atk)
        r2.metric("basis HP", basis_hp)
        r3.metric("옵션 토탈 보너스 ATK", opt_total_atk_bonus)
        r4.metric("옵션 토탈 보너스 HP", opt_total_hp_bonus)
        st.metric("옵션 토탈 전투력(환산)", int(_iround(opt_total_power)))
