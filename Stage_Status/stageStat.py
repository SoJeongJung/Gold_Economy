# stageStat.py
# Streamlit app: Stage clear combat power + power contributions (passive/agency/character/gear/slotLv/rune/options)
#
# New Spec 적용 + Passive ATK/HP 자동 추정(갭 최소화) 추가
#
# 실행:
#   python3 -m streamlit run stageStat.py

import json
import ast
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

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


def _floor(x: float) -> int:
    # spec: 소수점 내림
    try:
        return int(np.floor(float(x)))
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


def power_from_atk_hp(atk: float, hp: float, atk_weight: float = 4.0, hp_weight: float = 1.0) -> float:
    return atk * atk_weight + hp * hp_weight


@dataclass
class ItemRow:
    id: str
    name: str
    slotType: str
    atkBase: float
    atkInc: float
    hpBase: float
    hpInc: float
    optionAtk: int          # % (예: 5 -> 5%)
    optionHp: int
    optionAtkself: int      # % (예: 20 -> 20%)
    optionHpself: int


def load_item_lookup(item_file) -> Dict[str, ItemRow]:
    df = pd.read_csv(item_file)
    df = _clean_cols(df)

    id_col = _get_col(df, "id")
    name_col = _get_col(df, "name")
    slot_col = _get_col(df, "slotType", "slottype", "slot_type")
    if id_col is None or name_col is None or slot_col is None:
        raise ValueError("Item lookup CSV must include: id, name, slotType")

    for c in ["atkBase", "atkInc", "hpBase", "hpInc"]:
        cc = _get_col(df, c)
        if cc is None:
            df[c] = 0
        else:
            df[c] = pd.to_numeric(df[cc], errors="coerce").fillna(0)

    for c in ["optionAtk", "optionHp", "optionAtkself", "optionHpself"]:
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
            optionAtkself=int(r["optionAtkself"]),
            optionHpself=int(r["optionHpself"]),
        )
    return out


def load_agency_lookup(agency_file) -> Dict[int, Tuple[float, float]]:
    """
    CSV columns: agencyLv, agencyAtk, agencyHp
    returns {agencyLv: (agencyAtk, agencyHp)}
    """
    df = pd.read_csv(agency_file)
    df = _clean_cols(df)

    lvl_col = _get_col(df, "agencyLv", "lvl", "level", "agency_lv")
    atk_col = _get_col(df, "agencyAtk", "atk", "agency_atk")
    hp_col = _get_col(df, "agencyHp", "hp", "agency_hp", "agnecyHp", "agnecyhp")  # typo tolerance

    if lvl_col is None or atk_col is None or hp_col is None:
        raise ValueError("Agency lookup CSV must include columns: agencyLv, agencyAtk, agencyHp")

    df["lvl_i"] = df[lvl_col].apply(_to_int)
    df["atk_f"] = df[atk_col].apply(_to_float)
    df["hp_f"] = df[hp_col].apply(_to_float)

    return dict(zip(df["lvl_i"], list(zip(df["atk_f"], df["hp_f"]))))


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


def build_power_long_df(df: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, List[str]]:
    series_cols = [
        ("전체 전투력", "total_power"),
        ("장비 전투력", "gear_power_total_for_plot"),
        ("룬 전투력", "rune_power_total_for_plot"),
        ("장비 레벨 전투력", "slotLv_power"),
        ("에이전시 레벨 전투력", "agency_power"),
        ("그 외 옵션 전투력", "other_option_power"),
    ]

    rows = []
    for label, col in series_cols:
        if col not in df.columns:
            continue
        tmp = df[["stage_lv"]].copy()
        tmp["series"] = label
        tmp["value"] = df[col].astype(float)
        if mode != "avg":
            tmp["user"] = df["user"].astype(str)
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(), [c for _, c in series_cols]

    out = pd.concat(rows, ignore_index=True)

    if mode != "avg":
        last_stage = int(df["stage_lv"].max()) if len(df) else 0
        ranking = (
            df[df["stage_lv"] == last_stage]
            .groupby("user")["total_power"]
            .max()
            .sort_values(ascending=False)
        )
        user_order = ranking.index.tolist()
        out["user"] = pd.Categorical(out["user"], categories=user_order, ordered=True)

    return out, [label for label, _ in series_cols]


def _gap_label(g: float) -> str:
    gi = int(_floor(g))
    if gi == 0:
        return "정확"
    if gi > 0:
        return f"부족 {gi}"
    return f"초과 {abs(gi)}"


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
        "룩업 CSV 2 (Items: Equip+Rune) — 컬럼: id,name,slotType,atkBase,atkInc,hpBase,hpInc,optionAtk,optionHp,optionAtkself,optionHpself",
        type=["csv"],
        accept_multiple_files=False,
    )

    st.divider()
    st.subheader("Character 고정값")
    character_atk = st.number_input("characterAtk", min_value=0, max_value=10_000, value=75, step=1)
    character_hp = st.number_input("characterHp", min_value=0, max_value=100_000, value=300, step=10)

    st.divider()
    st.subheader("Passive ATK/HP (수동/자동)")
    auto_fit_passive = st.checkbox("Passive 자동 보정(갭 최소화)", value=True)
    fit_step = st.selectbox("탐색 step", options=[0.001, 0.002, 0.005, 0.01, 0.02], index=3)
    fit_metric = st.selectbox("목적함수", options=["MAE(|gap| 평균)", "RMSE(gap 제곱 평균)"], index=0)

    st.caption("자동 보정 OFF일 때만 수동 비율을 사용합니다.")
    passive_atk_ratio_manual = st.slider("passiveAtk 비율(수동)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

    st.divider()
    st.subheader("slotType 매핑")
    st.caption("Item.slotType 문자열 → 유저 CSV의 slot_type(1~6)로 매핑 (기본: Hat/Coat/Ring/Neck/Belt/Boots)")
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
    st.info("왼쪽에서 Item 룩업 CSV를 업로드하세요.")
    st.stop()

try:
    agency_map = load_agency_lookup(agency_lookup_file)
    item_map = load_item_lookup(item_lookup_file)
except Exception as e:
    st.error(f"룩업 파일 로딩 실패: {e}")
    st.stop()


# =========================
# Core calc per snapshot
# =========================
def compute_components_for_snapshot(
    equips: List[str],
    runes: List[str],
    slot_lv_by_type_id: Dict[int, int],
    agency_lv: int,
    final_power: float,
    passive_atk_ratio: float,
) -> Dict[str, Any]:
    passive_hp_ratio = 1.0 - float(passive_atk_ratio)

    # ---------- agency
    agency_atk, agency_hp = agency_map.get(int(agency_lv), (0.0, 0.0))
    agency_power = power_from_atk_hp(agency_atk, agency_hp)

    # ---------- character (fixed)
    character_atk_f = float(character_atk)
    character_hp_f = float(character_hp)
    character_power = power_from_atk_hp(character_atk_f, character_hp_f)

    # ---------- gear base + slotLv contribution + gear options%
    gear_base_atk = 0.0
    gear_base_hp = 0.0
    slotLv_atk = 0.0
    slotLv_hp = 0.0

    gear_option_pct_atk = 0.0
    gear_option_pct_hp = 0.0
    gear_option_self_atk = 0.0
    gear_option_self_hp = 0.0

    equip_names: List[str] = []

    for eid in equips:
        it = item_map.get(str(eid))
        if not it:
            equip_names.append(f"(missing:{eid})")
            continue
        equip_names.append(it.name)

        gear_base_atk += it.atkBase
        gear_base_hp += it.hpBase

        type_id = slotType_to_type_id.get(it.slotType)
        lv = slot_lv_by_type_id.get(int(type_id), 0) if type_id is not None else 0

        slotLv_atk += it.atkInc * lv
        slotLv_hp += it.hpInc * lv

        gear_option_pct_atk += (it.optionAtk / 100.0)
        gear_option_pct_hp += (it.optionHp / 100.0)

        equip_atk_total = it.atkBase + it.atkInc * lv
        equip_hp_total = it.hpBase + it.hpInc * lv
        if it.optionAtkself:
            gear_option_self_atk += equip_atk_total * (it.optionAtkself / 100.0)
        if it.optionHpself:
            gear_option_self_hp += equip_hp_total * (it.optionHpself / 100.0)

    gear_power = power_from_atk_hp(gear_base_atk, gear_base_hp)
    slotLv_power = power_from_atk_hp(slotLv_atk, slotLv_hp)

    # ---------- runes base + rune options%
    rune_base_atk = 0.0
    rune_base_hp = 0.0

    rune_option_pct_atk = 0.0
    rune_option_pct_hp = 0.0
    rune_option_self_atk = 0.0
    rune_option_self_hp = 0.0

    rune_names: List[str] = []

    for rid in runes:
        it = item_map.get(str(rid))
        if not it:
            rune_names.append(f"(missing:{rid})")
            continue
        rune_names.append(it.name)

        rune_base_atk += it.atkBase
        rune_base_hp += it.hpBase

        rune_option_pct_atk += (it.optionAtk / 100.0)
        rune_option_pct_hp += (it.optionHp / 100.0)

        if it.optionAtkself:
            rune_option_self_atk += it.atkBase * (it.optionAtkself / 100.0)
        if it.optionHpself:
            rune_option_self_hp += it.hpBase * (it.optionHpself / 100.0)

    rune_power = power_from_atk_hp(rune_base_atk, rune_base_hp)

    # ---------- passive initial (exclude options first)
    base_sum_power_excl_options = agency_power + character_power + gear_power + slotLv_power + rune_power
    passive_power0 = float(final_power) - float(base_sum_power_excl_options)

    # passive split heuristic (only know power, not atk/hp)
    # We split in a consistent way: passive_power0 is allocated into atk-space and hp-space by ratio.
    # atk-part is converted back to "atk" using /4.
    passive_atk0 = (passive_power0 * passive_atk_ratio) / 4.0
    passive_hp0 = (passive_power0 * passive_hp_ratio)

    # ---------- option base sums (spec includes passive/agency/character/gear/rune)
    base_atk_for_option = passive_atk0 + agency_atk + character_atk_f + gear_base_atk + rune_base_atk
    base_hp_for_option = passive_hp0 + agency_hp + character_hp_f + gear_base_hp + rune_base_hp

    # total options on base
    gear_option_total_atk = base_atk_for_option * gear_option_pct_atk
    gear_option_total_hp = base_hp_for_option * gear_option_pct_hp
    rune_option_total_atk = base_atk_for_option * rune_option_pct_atk
    rune_option_total_hp = base_hp_for_option * rune_option_pct_hp

    # add self options
    gear_option_atk = gear_option_total_atk + gear_option_self_atk
    gear_option_hp = gear_option_total_hp + gear_option_self_hp
    rune_option_atk = rune_option_total_atk + rune_option_self_atk
    rune_option_hp = rune_option_total_hp + rune_option_self_hp

    gear_option_power = power_from_atk_hp(gear_option_atk, gear_option_hp)
    rune_option_power = power_from_atk_hp(rune_option_atk, rune_option_hp)

    # ---------- passive final (subtract all options too)
    calc_sum = base_sum_power_excl_options + gear_option_power + rune_option_power
    passive_power = float(final_power) - float(calc_sum)

    other_option_power = passive_power

    out = {
        "total_power": _floor(final_power),

        "agency_lv": int(agency_lv),
        "agency_atk": _floor(agency_atk),
        "agency_hp": _floor(agency_hp),
        "agency_power": _floor(agency_power),

        "character_atk": _floor(character_atk_f),
        "character_hp": _floor(character_hp_f),
        "character_power": _floor(character_power),

        "gear_atk": _floor(gear_base_atk),
        "gear_hp": _floor(gear_base_hp),
        "gear_power": _floor(gear_power),

        "slotLv_atk": _floor(slotLv_atk),
        "slotLv_hp": _floor(slotLv_hp),
        "slotLv_power": _floor(slotLv_power),

        "rune_atk": _floor(rune_base_atk),
        "rune_hp": _floor(rune_base_hp),
        "rune_power": _floor(rune_power),

        "gear_option_pct_atk_sum": gear_option_pct_atk,
        "gear_option_pct_hp_sum": gear_option_pct_hp,
        "gear_option_total_atk": _floor(gear_option_total_atk),
        "gear_option_total_hp": _floor(gear_option_total_hp),
        "gear_option_self_atk": _floor(gear_option_self_atk),
        "gear_option_self_hp": _floor(gear_option_self_hp),
        "gear_option_power": _floor(gear_option_power),

        "rune_option_pct_atk_sum": rune_option_pct_atk,
        "rune_option_pct_hp_sum": rune_option_pct_hp,
        "rune_option_total_atk": _floor(rune_option_total_atk),
        "rune_option_total_hp": _floor(rune_option_total_hp),
        "rune_option_self_atk": _floor(rune_option_self_atk),
        "rune_option_self_hp": _floor(rune_option_self_hp),
        "rune_option_power": _floor(rune_option_power),

        "passive_power": _floor(passive_power),
        "other_option_power": _floor(other_option_power),

        "option_base_atk": _floor(base_atk_for_option),
        "option_base_hp": _floor(base_hp_for_option),

        "equips_names": equip_names,
        "runes_names": rune_names,
    }

    out["gear_power_total_for_plot"] = _floor(out["gear_power"] + out["gear_option_power"])
    out["rune_power_total_for_plot"] = _floor(out["rune_power"] + out["rune_option_power"])

    out["calc_sum_power"] = _floor(
        out["agency_power"]
        + out["character_power"]
        + out["gear_power"]
        + out["slotLv_power"]
        + out["rune_power"]
        + out["gear_option_power"]
        + out["rune_option_power"]
        + out["passive_power"]
    )
    out["gap_total_minus_calc"] = _floor(out["total_power"] - out["calc_sum_power"])
    out["gap_label"] = _gap_label(out["gap_total_minus_calc"])

    return out


# =========================
# Build per-user snapshots (raw extraction first)
# =========================
def build_raw_snapshots_for_user(df: pd.DataFrame, user_label: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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

            equips = [str(x) for x in (gs.get("equips", []) or [])]
            runes = [str(x) for x in (gs.get("runes", []) or [])]

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

            row = {
                "user": user_label,
                "time": float(r.get("time", 0.0) or 0.0),
                "stage_lv": int(r.get("stage_lv", 0) or 0),
                "stage_id": str(r.get("stage_id", "")),
                "total_power": float(r.get("combat_power", 0.0) or 0.0),
                "agency_lv": int(agency_lv_current),
                "equips_ids": equips,
                "runes_ids": runes,
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


def build_breakdown_tables_for_snapshot(
    snapshot_row: pd.Series,
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
                "atkBase": 0.0,
                "hpBase": 0.0,
                "atkInc": 0.0,
                "hpInc": 0.0,
                "slotLv_atk": 0.0,
                "slotLv_hp": 0.0,
                "base_power": 0.0,
                "slotLv_power": 0.0,
                "optionAtk%": 0,
                "optionHp%": 0,
                "optionAtkself%": 0,
                "optionHpself%": 0,
            })
            continue

        type_id = slotType_to_type_id.get(it.slotType)
        lv = slot_lv_by_type_id.get(int(type_id), 0) if type_id is not None else 0

        slot_atk = it.atkInc * lv
        slot_hp = it.hpInc * lv

        equip_rows.append({
            "name": it.name,
            "equip_id": it.id,
            "slotType": it.slotType,
            "slot_type_id": type_id,
            "slot_lv": lv,
            "atkBase": it.atkBase,
            "hpBase": it.hpBase,
            "atkInc": it.atkInc,
            "hpInc": it.hpInc,
            "slotLv_atk": slot_atk,
            "slotLv_hp": slot_hp,
            "base_power": power_from_atk_hp(it.atkBase, it.hpBase),
            "slotLv_power": power_from_atk_hp(slot_atk, slot_hp),
            "optionAtk%": it.optionAtk,
            "optionHp%": it.optionHp,
            "optionAtkself%": it.optionAtkself,
            "optionHpself%": it.optionHpself,
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
                "atkBase": 0.0,
                "hpBase": 0.0,
                "base_power": 0.0,
                "optionAtk%": 0,
                "optionHp%": 0,
                "optionAtkself%": 0,
                "optionHpself%": 0,
            })
            continue

        rune_rows.append({
            "name": it.name,
            "rune_id": it.id,
            "slotType": it.slotType,
            "atkBase": it.atkBase,
            "hpBase": it.hpBase,
            "base_power": power_from_atk_hp(it.atkBase, it.hpBase),
            "optionAtk%": it.optionAtk,
            "optionHp%": it.optionHp,
            "optionAtkself%": it.optionAtkself,
            "optionHpself%": it.optionHpself,
        })

    rune_df = pd.DataFrame(rune_rows)
    return slot_levels_df, equip_df, rune_df


# =========================
# Passive auto-fit (1D grid search)
# =========================
@st.cache_data(show_spinner=False)
def fit_best_passive_ratio(df_raw: pd.DataFrame, step: float, metric: str) -> Tuple[float, float]:
    """
    Returns (best_ratio, best_score)
    metric: "MAE(|gap| 평균)" or "RMSE(gap 제곱 평균)"
    """
    if df_raw.empty:
        return 0.5, float("inf")

    # Pre-extract inputs to avoid pandas overhead
    equips_list = df_raw["equips_ids"].tolist()
    runes_list = df_raw["runes_ids"].tolist()
    agency_lv_list = df_raw["agency_lv"].astype(int).tolist()
    final_power_list = df_raw["total_power"].astype(float).tolist()

    slot_cols = [f"slot_lv_{i}" for i in range(1, 7)]
    slot_mat = df_raw[slot_cols].fillna(0).astype(int).to_numpy()

    ratios = np.arange(0.0, 1.0 + step/2, step)
    best_ratio = 0.5
    best_score = float("inf")

    for r in ratios:
        gaps = []
        for idx in range(len(df_raw)):
            slot_lv_by_type = {i+1: int(slot_mat[idx, i]) for i in range(6)}
            comp = compute_components_for_snapshot(
                equips=equips_list[idx],
                runes=runes_list[idx],
                slot_lv_by_type_id=slot_lv_by_type,
                agency_lv=int(agency_lv_list[idx]),
                final_power=float(final_power_list[idx]),
                passive_atk_ratio=float(r),
            )
            gaps.append(float(comp["gap_total_minus_calc"]))

        g = np.array(gaps, dtype=float)
        if metric.startswith("MAE"):
            score = float(np.mean(np.abs(g)))
        else:
            score = float(np.sqrt(np.mean(g * g)))

        if score < best_score:
            best_score = score
            best_ratio = float(r)

    return best_ratio, best_score


def apply_components(df_raw: pd.DataFrame, ratio: float) -> pd.DataFrame:
    # Recompute all component columns for the given ratio
    out_rows = []
    for _, row in df_raw.iterrows():
        slot_lv_by_type = {i: int(row.get(f"slot_lv_{i}", 0) or 0) for i in range(1, 7)}
        comp = compute_components_for_snapshot(
            equips=row.get("equips_ids", []) or [],
            runes=row.get("runes_ids", []) or [],
            slot_lv_by_type_id=slot_lv_by_type,
            agency_lv=int(row.get("agency_lv", 0) or 0),
            final_power=float(row.get("total_power", 0.0) or 0.0),
            passive_atk_ratio=float(ratio),
        )
        merged = dict(row.to_dict())
        # rename: raw total_power is float; overwrite to floored int consistent
        merged.update(comp)
        merged["equips"] = names_join(comp["equips_names"])
        merged["runes"] = names_join(comp["runes_names"])
        out_rows.append(merged)
    return pd.DataFrame(out_rows)


# =========================
# Run pipeline
# =========================
all_raw = []
validations = []
errors = []

for f in user_files:
    user_label = f.name.replace(".csv", "")
    try:
        udf = parse_user_csv(f)
        raw, vinfo = build_raw_snapshots_for_user(udf, user_label=user_label)
        validations.append(vinfo)
        if not raw.empty:
            all_raw.append(raw)
        else:
            errors.append(f"{user_label}: first_clear game_end 스냅샷이 없습니다.")
    except Exception as e:
        errors.append(f"{user_label}: 처리 실패 ({e})")

if not all_raw:
    st.error("유효한 stage clear 데이터가 없습니다. (조건: game_end + game_state.first_clear == true)")
    if errors:
        with st.expander("처리 오류", expanded=True):
            for msg in errors:
                st.write(f"- {msg}")
    st.stop()

df_raw_all = pd.concat(all_raw, ignore_index=True)

# Stage range filter
df_raw_all = df_raw_all[(df_raw_all["stage_lv"] >= 1) & (df_raw_all["stage_lv"] <= int(max_stage))].copy()

# Decide ratio (auto-fit or manual)
if auto_fit_passive:
    with st.sidebar:
        st.subheader("Passive 자동 보정 결과")
        with st.spinner("PassiveAtk 비율 자동 추정 중..."):
            best_ratio, best_score = fit_best_passive_ratio(df_raw_all, float(fit_step), str(fit_metric))
        st.success(f"best passiveAtk ratio = {best_ratio:.3f}")
        st.write(f"score ({fit_metric}) = {best_score:.3f}")
    passive_atk_ratio_used = float(best_ratio)
else:
    passive_atk_ratio_used = float(passive_atk_ratio_manual)

# Compute components with chosen ratio
df_all = apply_components(df_raw_all, passive_atk_ratio_used)

# Keep ID strings too (debug)
df_all["equips_ids_str"] = df_all["equips_ids"].apply(names_join)
df_all["runes_ids_str"] = df_all["runes_ids"].apply(names_join)


# =========================
# UI: Tabs
# =========================
tab_overall, tab_user, tab_avg, tab_validate = st.tabs(["전체(그래프)", "유저별(그래프+검증)", "스테이지 평균", "데이터 오류 검증"])


# -------------------------
# Tab: Overall
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

    long_df, _ = build_power_long_df(view.sort_values(["user", "stage_lv"]), mode="overall")

    fig = px.line(
        long_df,
        x="stage_lv",
        y="value",
        color="user",
        line_group="series",
        facet_row="series",
        title="전체: Stage별 전투력 구성 (범례 클릭으로 토글)",
    )
    fig.update_layout(height=900)
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

    c1, c2 = st.columns([1, 2])
    with c1:
        show_points = st.checkbox("포인트 표시", value=True, key="user_points")
    with c2:
        stage_focus = st.selectbox(
            "상세 확인할 스테이지(선택)",
            options=udf["stage_lv"].tolist(),
            index=len(udf) - 1 if len(udf) > 0 else 0,
            key="user_stage_focus",
        )

    long_u, _ = build_power_long_df(udf, mode="overall")
    fig2 = px.line(
        long_u,
        x="stage_lv",
        y="value",
        color="series",
        markers=bool(show_points),
        title=f"{user_sel}: Stage별 전투력 구성 (범례 클릭으로 토글)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("선택 스테이지 상세 분해 (장비/룬/슬롯레벨/옵션 확인)")

    snap = udf[udf["stage_lv"] == int(stage_focus)].iloc[0]
    slot_levels_df, equip_df, rune_df = build_breakdown_tables_for_snapshot(snap)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("전체 전투력", f"{snap['total_power']}")
    m2.metric("에이전시", f"{snap['agency_power']}")
    m3.metric("캐릭터", f"{snap['character_power']}")
    m4.metric("장비(base)", f"{snap['gear_power']}")
    m5.metric("장비레벨", f"{snap['slotLv_power']}")
    m6.metric("룬(base)", f"{snap['rune_power']}")

    m7, m8, m9, m10, m11, m12 = st.columns(6)
    m7.metric("장비 옵션", f"{snap['gear_option_power']}")
    m8.metric("룬 옵션", f"{snap['rune_option_power']}")
    m9.metric("그 외 옵션(=passive)", f"{snap['other_option_power']}")
    m10.metric("합산(calc_sum)", f"{snap['calc_sum_power']}")
    m11.metric("갭", f"{snap['gap_total_minus_calc']}")
    m12.metric("갭 라벨", snap["gap_label"])

    st.caption(
        f"현재 적용 passiveAtkRatio = {passive_atk_ratio_used:.3f} (auto={auto_fit_passive}). "
        "옵션 베이스(ATK/HP) = passive(분해) + agency + character + gear(base) + rune(base)."
    )

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
            gear_power_total_for_plot=("gear_power_total_for_plot", "mean"),
            rune_power_total_for_plot=("rune_power_total_for_plot", "mean"),
            slotLv_power=("slotLv_power", "mean"),
            agency_power=("agency_power", "mean"),
            other_option_power=("other_option_power", "mean"),
        )
        .sort_values("stage_lv")
    )

    long_a, _ = build_power_long_df(agg, mode="avg")
    fig3 = px.line(
        long_a,
        x="stage_lv",
        y="value",
        color="series",
        title="스테이지 평균: 전투력 구성 (범례 클릭으로 토글)",
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
