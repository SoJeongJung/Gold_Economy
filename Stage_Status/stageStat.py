# stageStat.py
# Streamlit app: Stage clear combat power + power contributions
#
# ✅ 이번 수정(요청 반영)
# 1) item 룩업의 self 옵션 컬럼명 변경 대응
#    - optionAtkBase/optionHpBase 뿐 아니라 optionAtkself/optionHpself도 자동 인식
#    - 장비/룬 모두 self 옵션(=자기 자신 스탯 기준 %) 계산에 포함
#
# 2) game_end의 equips/runes를 더 “정확/강건”하게 파싱
#    - game_state.equips / game_state.runes 가
#      * [id, id, ...]
#      * [{"id":...}, ...]
#      * {"ids":[...]} 형태여도 id를 추출
#
# 3) agency 레벨/파워 누락 방지
#    - agency_lvup 이벤트만 보지 않고,
#      csv의 agency_lv 컬럼 값(>0)이 있으면 항상 최신으로 갱신
#    - 룩업 CSV1은 agencyLv, agencyAtk, agencyHp
#
# 4) 옵션 계산 시 사용되는 “가정 패시브 ATK/HP”를 스테이지에 따라 증가(10스테이지 단위)
#    - 기본 시작값: 75 / 300
#    - 최대값: 900 / 3600
#    - 80스테이지에서 최대값 도달
#    - 옵션 계산에만 사용(실제 passive_power는 최종 잔여로 계산)
#
# 5) passive/character 음수 방지
#    - characterAtk/Hp는 0 이상
#    - passive_power = max(0, total_power - non_passive_sum)
#
# UI는 기존 탭 구조/흐름을 유지하고, 검증/계산 정확도를 강화했습니다.

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


def names_join(xs: Any) -> str:
    if isinstance(xs, list):
        return ", ".join([str(x) for x in xs])
    if xs is None or (isinstance(xs, float) and np.isnan(xs)):
        return ""
    return str(xs)


def _mode_from_list_series(series: pd.Series) -> str:
    vals: List[str] = []
    for x in series:
        if isinstance(x, list):
            vals.extend([str(v) for v in x if str(v).strip() != ""])
    if not vals:
        return ""
    c = Counter(vals)
    return c.most_common(1)[0][0]


def _order_components_by_last_value(long_df: pd.DataFrame, label_col: str, x_col: str, y_col: str) -> List[str]:
    if long_df.empty:
        return []
    last_x = long_df[x_col].max()
    tmp = long_df[long_df[x_col] == last_x].groupby(label_col, as_index=False)[y_col].sum()
    tmp = tmp.sort_values(y_col, ascending=False)
    return tmp[label_col].tolist()


def _extract_id_list(v: Any) -> List[str]:
    """
    game_state.equips / game_state.runes가 다양한 형태일 수 있어 강건하게 id를 추출.
    허용 형태:
      - [123, 456]
      - ["123", "456"]
      - [{"id":123}, {"id":"456"}]
      - {"ids":[...]} / {"items":[{"id":...}]}
      - "123,456" 같은 문자열
    """
    if v is None:
        return []
    if isinstance(v, list):
        out = []
        for e in v:
            if isinstance(e, (int, np.integer)):
                out.append(str(int(e)))
            elif isinstance(e, str):
                s = e.strip()
                if s != "":
                    out.append(s)
            elif isinstance(e, dict):
                for k in ["id", "itemId", "item_id", "equip_id", "rune_id"]:
                    if k in e and e[k] is not None and str(e[k]).strip() != "":
                        out.append(str(e[k]).strip())
                        break
        return out
    if isinstance(v, dict):
        for k in ["ids", "equips", "runes", "items", "list"]:
            if k in v:
                return _extract_id_list(v[k])
        return []
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return []
        if s.startswith("[") or s.startswith("{"):
            parsed = _safe_json_loads(s)
            return _extract_id_list(parsed)
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip() != ""]
        return parts
    if isinstance(v, (int, np.integer)):
        return [str(int(v))]
    return []


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
    optionAtk: float      # ratio(0~1), "총합 기준 %" 옵션
    optionHp: float       # ratio(0~1)
    optionAtkSelf: float  # ratio(0~1), "자기 자신 스탯 기준 %" 옵션
    optionHpSelf: float   # ratio(0~1)
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

    # 총합 기준 옵션(%)
    for c in ["optionAtk", "optionHp"]:
        cc = _get_col(df, c)
        if cc is None:
            df[c] = 0.0
        else:
            df[c] = df[cc].apply(_to_ratio_percent)

    # ✅ self 옵션 컬럼명: optionAtkBase/optionHpBase OR optionAtkself/optionHpself
    atk_self_col = _get_col(df, "optionAtkBase", "optionAtkself", "optionAtkSelf", "option_atk_self")
    hp_self_col = _get_col(df, "optionHpBase", "optionHpself", "optionHpSelf", "option_hp_self")

    if atk_self_col is None:
        df["optionAtkSelf"] = 0.0
    else:
        df["optionAtkSelf"] = df[atk_self_col].apply(_to_ratio_percent)

    if hp_self_col is None:
        df["optionHpSelf"] = 0.0
    else:
        df["optionHpSelf"] = df[hp_self_col].apply(_to_ratio_percent)

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
            optionAtkSelf=float(r["optionAtkSelf"]),
            optionHpSelf=float(r["optionHpSelf"]),
            gradeType=str(r["gradeType"]),
        )
    return out


def load_agency_lookup(agency_file) -> Dict[int, AgencyRow]:
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


def assumed_passive_by_stage(
    stage_lv: int,
    start_atk: float,
    start_hp: float,
    max_atk: float,
    max_hp: float,
    max_stage: int,
    step_stage: int,
) -> Tuple[float, float]:
    """
    옵션 계산 시에만 쓰는 가정 passive ATK/HP.
    - 10스테이지 단위(step_stage)로 증가
    - max_stage(기본 80)에서 max_atk/max_hp 도달
    """
    if step_stage <= 0:
        step_stage = 10
    if max_stage <= step_stage:
        max_stage = step_stage

    stage_c = max(1, int(stage_lv))
    stage_c = min(stage_c, int(max_stage))

    steps = int(max_stage // step_stage)  # 80/10=8
    if steps < 2:
        return float(max_atk), float(max_hp)

    idx = min((stage_c - 1) // step_stage, steps - 1)  # 0..7
    frac = idx / (steps - 1)  # 0..1

    atk = float(start_atk) + (float(max_atk) - float(start_atk)) * frac
    hp = float(start_hp) + (float(max_hp) - float(start_hp)) * frac
    return atk, hp


def sum_equip_rune_components(
    equip_ids: List[str],
    rune_ids: List[str],
    slot_lv_by_type_id: Dict[int, int],
    slotType_to_type_id: Dict[str, int],
    item_map: Dict[str, ItemRow],
    agencyAtk: float,
    agencyHp: float,
    characterAtk: float,
    characterHp: float,
    assumed_passiveAtk_for_option: float,
    assumed_passiveHp_for_option: float,
) -> Dict[str, Any]:
    """
    반환:
      - gear_base_power, slotLv_power, rune_base_power
      - gear_option_power(= gear_total + gear_self), rune_option_power(= rune_total + rune_self)
      - equips/runes names + grades
    """
    gear_base_power = 0.0
    slotLv_power = 0.0
    rune_base_power = 0.0

    gear_atk_sum = 0.0
    gear_hp_sum = 0.0
    rune_atk_sum = 0.0
    rune_hp_sum = 0.0

    equips_names: List[str] = []
    runes_names: List[str] = []
    equips_grades: List[str] = []
    runes_grades: List[str] = []

    # 옵션 퍼센트 합(gear / rune) — "총합 기준 %"
    gear_optAtk_pct_sum = 0.0
    gear_optHp_pct_sum = 0.0
    rune_optAtk_pct_sum = 0.0
    rune_optHp_pct_sum = 0.0

    # self 옵션 — "자기 자신 스탯 기준 %"
    option_gear_self_atk = 0.0
    option_gear_self_hp = 0.0
    option_rune_self_atk = 0.0
    option_rune_self_hp = 0.0

    # ---- equips
    for eid in equip_ids:
        it = item_map.get(str(eid))
        if not it:
            equips_names.append(f"(missing:{eid})")
            equips_grades.append("")
            continue

        equips_names.append(it.name)
        equips_grades.append(it.gradeType or "")

        type_id = slotType_to_type_id.get(it.slotType)
        lv = slot_lv_by_type_id.get(int(type_id), 0) if type_id is not None else 0

        # base: slot_lv=0 기준
        base_atk, base_hp, base_p = calc_item_power(it, 0, is_equip=True)
        # with slot
        atk, hp, p = calc_item_power(it, lv, is_equip=True)

        gear_base_power += base_p
        slotLv_power += (p - base_p)

        gear_atk_sum += atk
        gear_hp_sum += hp

        gear_optAtk_pct_sum += it.optionAtk
        gear_optHp_pct_sum += it.optionHp

        if it.optionAtkSelf:
            option_gear_self_atk += atk * it.optionAtkSelf
        if it.optionHpSelf:
            option_gear_self_hp += hp * it.optionHpSelf

    # ---- runes
    for rid in rune_ids:
        it = item_map.get(str(rid))
        if not it:
            runes_names.append(f"(missing:{rid})")
            runes_grades.append("")
            continue

        runes_names.append(it.name)
        runes_grades.append(it.gradeType or "")

        atk, hp, p = calc_item_power(it, 0, is_equip=False)

        rune_base_power += p
        rune_atk_sum += atk
        rune_hp_sum += hp

        rune_optAtk_pct_sum += it.optionAtk
        rune_optHp_pct_sum += it.optionHp

        # ✅ 룬도 self 옵션이 존재할 수 있으니 반영
        if it.optionAtkSelf:
            option_rune_self_atk += atk * it.optionAtkSelf
        if it.optionHpSelf:
            option_rune_self_hp += hp * it.optionHpSelf

    # 옵션 base (ATK/HP) — passive는 "가정값"으로 옵션에만 사용
    option_base_atk = (
        float(assumed_passiveAtk_for_option)
        + float(agencyAtk)
        + float(characterAtk)
        + float(gear_atk_sum)
        + float(rune_atk_sum)
    )
    option_base_hp = (
        float(assumed_passiveHp_for_option)
        + float(agencyHp)
        + float(characterHp)
        + float(gear_hp_sum)
        + float(rune_hp_sum)
    )

    # 총합 기준 옵션
    option_gear_total_atk = option_base_atk * gear_optAtk_pct_sum
    option_gear_total_hp = option_base_hp * gear_optHp_pct_sum

    option_rune_total_atk = option_base_atk * rune_optAtk_pct_sum
    option_rune_total_hp = option_base_hp * rune_optHp_pct_sum

    # power 환산
    option_gear_total_power = option_gear_total_atk * 4.0 + option_gear_total_hp
    option_gear_self_power = (option_gear_self_atk * 4.0) + option_gear_self_hp

    option_rune_total_power = option_rune_total_atk * 4.0 + option_rune_total_hp
    option_rune_self_power = (option_rune_self_atk * 4.0) + option_rune_self_hp

    gear_option_power = option_gear_total_power + option_gear_self_power
    rune_option_power = option_rune_total_power + option_rune_self_power

    return {
        "gear_base_power": gear_base_power,
        "slotLv_power": slotLv_power,
        "rune_base_power": rune_base_power,

        "gear_option_power": gear_option_power,
        "rune_option_power": rune_option_power,

        "option_gear_total_power": option_gear_total_power,
        "option_gear_self_power": option_gear_self_power,
        "option_rune_total_power": option_rune_total_power,
        "option_rune_self_power": option_rune_self_power,

        "gear_atk_sum": gear_atk_sum,
        "gear_hp_sum": gear_hp_sum,
        "rune_atk_sum": rune_atk_sum,
        "rune_hp_sum": rune_hp_sum,

        "option_base_atk": option_base_atk,
        "option_base_hp": option_base_hp,

        "equips_names": equips_names,
        "runes_names": runes_names,
        "equips_grades": equips_grades,
        "runes_grades": runes_grades,

        "gear_optAtk_pct_sum": gear_optAtk_pct_sum,
        "gear_optHp_pct_sum": gear_optHp_pct_sum,
        "rune_optAtk_pct_sum": rune_optAtk_pct_sum,
        "rune_optHp_pct_sum": rune_optHp_pct_sum,
    }


# =========================
# Build snapshots
# =========================
def build_stage_rows_for_user(
    df: pd.DataFrame,
    user_label: str,
    agency_map: Dict[int, AgencyRow],
    slotType_to_type_id: Dict[str, int],
    item_map: Dict[str, ItemRow],
    characterAtk: float,
    characterHp: float,
    passive_start_atk: float,
    passive_start_hp: float,
    passive_max_atk: float,
    passive_max_hp: float,
    passive_max_stage: int,
    passive_step_stage: int,
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
        "passive_clamped_rows": 0,
    }

    for _, r in df.iterrows():
        ev = str(r["event_name"])

        # ✅ agency_lvup만 보지 말고, 컬럼에 값 있으면 항상 반영
        alv_any = int(r.get("agency_lv", 0) or 0)
        if alv_any > 0:
            agency_lv_current = alv_any

        # ✅ slot도 이벤트명에 덜 의존(컬럼이 유효하면 반영)
        stype_any = int(r.get("slot_type", 0) or 0)
        slv_any = int(r.get("slot_lv", 0) or 0)
        if stype_any > 0:
            slot_lv_by_type_id[stype_any] = slv_any

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

            # ✅ equips/runes는 반드시 game_state에서 가져옴(요구사항)
            equips = _extract_id_list(gs.get("equips"))
            runes = _extract_id_list(gs.get("runes"))

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

            # character (음수 방지)
            cAtk = max(0.0, float(characterAtk))
            cHp = max(0.0, float(characterHp))
            character_power = cAtk * 4.0 + cHp

            # ✅ 옵션 계산용 가정 passive atk/hp: 스테이지 기반 증가
            assumed_p_atk, assumed_p_hp = assumed_passive_by_stage(
                stage_lv=stage_lv,
                start_atk=passive_start_atk,
                start_hp=passive_start_hp,
                max_atk=passive_max_atk,
                max_hp=passive_max_hp,
                max_stage=passive_max_stage,
                step_stage=passive_step_stage,
            )

            comps = sum_equip_rune_components(
                equip_ids=equips,
                rune_ids=runes,
                slot_lv_by_type_id=slot_lv_by_type_id,
                slotType_to_type_id=slotType_to_type_id,
                item_map=item_map,
                agencyAtk=agencyAtk,
                agencyHp=agencyHp,
                characterAtk=cAtk,
                characterHp=cHp,
                assumed_passiveAtk_for_option=assumed_p_atk,
                assumed_passiveHp_for_option=assumed_p_hp,
            )

            non_passive_sum = (
                agency_power
                + character_power
                + comps["gear_base_power"]
                + comps["slotLv_power"]
                + comps["rune_base_power"]
                + comps["gear_option_power"]
                + comps["rune_option_power"]
            )

            passive_raw = float(total_power) - float(non_passive_sum)

            # 음수 방지: passive는 0으로 클램프
            passive_power = passive_raw
            if passive_power < 0:
                passive_power = 0.0
                validation["passive_clamped_rows"] += 1

            calc_sum_power = non_passive_sum + passive_power

            row = {
                "user": user_label,
                "time": float(r.get("time", 0.0) or 0.0),
                "stage_lv": stage_lv,
                "stage_id": str(r.get("stage_id", "")),
                "total_power": total_power,

                # agency
                "agency_lv": int(agency_lv_current),
                "agencyAtk": agencyAtk,
                "agencyHp": agencyHp,
                "agency_power": agency_power,

                # character
                "characterAtk": cAtk,
                "characterHp": cHp,
                "character_power": character_power,

                # gear/rune/slot + options
                "gear_base_power": comps["gear_base_power"],
                "slotLv_power": comps["slotLv_power"],
                "rune_base_power": comps["rune_base_power"],

                "gear_option_power": comps["gear_option_power"],
                "rune_option_power": comps["rune_option_power"],

                "option_gear_total_power": comps["option_gear_total_power"],
                "option_gear_self_power": comps["option_gear_self_power"],
                "option_rune_total_power": comps["option_rune_total_power"],
                "option_rune_self_power": comps["option_rune_self_power"],

                "gear_atk_sum": comps["gear_atk_sum"],
                "gear_hp_sum": comps["gear_hp_sum"],
                "rune_atk_sum": comps["rune_atk_sum"],
                "rune_hp_sum": comps["rune_hp_sum"],
                "option_base_atk": comps["option_base_atk"],
                "option_base_hp": comps["option_base_hp"],

                "gear_optAtk_pct_sum": comps["gear_optAtk_pct_sum"],
                "gear_optHp_pct_sum": comps["gear_optHp_pct_sum"],
                "rune_optAtk_pct_sum": comps["rune_optAtk_pct_sum"],
                "rune_optHp_pct_sum": comps["rune_optHp_pct_sum"],

                # 옵션 계산용 가정 passive
                "assumed_passiveAtk_for_option": assumed_p_atk,
                "assumed_passiveHp_for_option": assumed_p_hp,

                "equips_ids": equips,
                "runes_ids": runes,
                "equips_names": comps["equips_names"],
                "runes_names": comps["runes_names"],
                "equips_grades": comps["equips_grades"],
                "runes_grades": comps["runes_grades"],

                # sums
                "non_passive_sum": non_passive_sum,
                "passive_power": passive_power,
                "calc_sum_power": calc_sum_power,
                "passive_raw": passive_raw,  # 디버그용(표에서 숨길 수 있음)
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
                "atk(with slot)": 0.0,
                "hp(with slot)": 0.0,
                "power(with slot)": 0.0,
                "atk(slot_lv=0)": 0.0,
                "hp(slot_lv=0)": 0.0,
                "base_power(slot_lv=0)": 0.0,
                "slotLv_power": 0.0,
                "optionAtk(ratio)": 0.0,
                "optionHp(ratio)": 0.0,
                "optionAtkself(ratio)": 0.0,
                "optionHpself(ratio)": 0.0,
            })
            continue

        type_id = slotType_to_type_id.get(it.slotType)
        lv = slot_lv_by_type_id.get(int(type_id), 0) if type_id is not None else 0

        atk, hp, p = calc_item_power(it, lv, is_equip=True)
        base_atk, base_hp, base_p = calc_item_power(it, 0, is_equip=True)

        equip_rows.append({
            "name": it.name,
            "equip_id": it.id,  # 검증용 유지
            "gradeType": it.gradeType,
            "slotType": it.slotType,
            "slot_type_id": type_id,
            "slot_lv": lv,
            "atk(with slot)": atk,
            "hp(with slot)": hp,
            "power(with slot)": p,
            "atk(slot_lv=0)": base_atk,
            "hp(slot_lv=0)": base_hp,
            "base_power(slot_lv=0)": base_p,
            "slotLv_power": p - base_p,
            "optionAtk(ratio)": it.optionAtk,
            "optionHp(ratio)": it.optionHp,
            "optionAtkself(ratio)": it.optionAtkSelf,
            "optionHpself(ratio)": it.optionHpSelf,
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
                "base_power": 0.0,
                "optionAtk(ratio)": 0.0,
                "optionHp(ratio)": 0.0,
                "optionAtkself(ratio)": 0.0,
                "optionHpself(ratio)": 0.0,
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
            "base_power": p,
            "optionAtk(ratio)": it.optionAtk,
            "optionHp(ratio)": it.optionHp,
            "optionAtkself(ratio)": it.optionAtkSelf,
            "optionHpself(ratio)": it.optionHpSelf,
        })

    rune_df = pd.DataFrame(rune_rows)
    return slot_levels_df, equip_df, rune_df


# =========================
# Graph helpers
# =========================
LABELS = {
    "total_power": "전체 전투력",
    "passive_power": "패시브",
    "agency_power": "에이전시",
    "character_power": "캐릭터",
    "gear_power": "장비",
    "rune_power": "룬",
    "slotLv_power": "슬롯 레벨",
    "gear_option_power": "(장비 옵션)",
    "rune_option_power": "(룬 옵션)",
    "gear_base_power": "장비(옵션 제외)",
    "rune_base_power": "룬(옵션 제외)",
}


def build_stage_agg(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "total_power",
        "passive_power", "agency_power", "character_power",
        "gear_base_power", "slotLv_power", "rune_base_power",
        "gear_option_power", "rune_option_power",
        "calc_sum_power",
    ]
    tmp = df.copy()
    for c in cols:
        if c not in tmp.columns:
            tmp[c] = 0.0

    stage_agg = (
        tmp.groupby("stage_lv", as_index=False)[cols]
        .mean()
        .sort_values("stage_lv")
    )
    return stage_agg


def build_long_for_plot(stage_agg: pd.DataFrame, split_options: bool, include_total: bool, mode: str) -> pd.DataFrame:
    df = stage_agg.copy()

    # merged powers (6개 보기)
    df["gear_power"] = df["gear_base_power"].astype(float) + df["gear_option_power"].astype(float)
    df["rune_power"] = df["rune_base_power"].astype(float) + df["rune_option_power"].astype(float)

    if split_options:
        comp_cols = [
            "passive_power", "agency_power", "character_power",
            "gear_base_power", "rune_base_power", "slotLv_power",
            "gear_option_power", "rune_option_power",
        ]
    else:
        comp_cols = [
            "passive_power", "agency_power", "character_power",
            "gear_power", "rune_power", "slotLv_power",
        ]

    plot_cols = comp_cols.copy()
    if include_total:
        plot_cols = ["total_power"] + plot_cols

    long = df[["stage_lv"] + plot_cols].melt(id_vars=["stage_lv"], var_name="component", value_name="value")
    long["label"] = long["component"].map(LABELS).fillna(long["component"])

    if mode == "pct":
        if include_total:
            long = long[long["component"] != "total_power"].copy()

    return long


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
        "룩업 CSV 1 (Agency) — agencyLv, agencyAtk, agencyHp",
        type=["csv"],
        accept_multiple_files=False,
    )

    item_lookup_file = st.file_uploader(
        "룩업 CSV 2 (Items) — id,name,slotType,atkBase,atkInc,hpBase,hpInc,option...,gradeType",
        type=["csv"],
        accept_multiple_files=False,
    )

    st.divider()
    st.subheader("캐릭터(고정값)")
    characterAtk = st.number_input("characterAtk", min_value=0.0, value=75.0, step=1.0)
    characterHp = st.number_input("characterHp", min_value=0.0, value=300.0, step=10.0)

    with st.expander("옵션 계산용 패시브 ATK/HP (스테이지 기반 증가)", expanded=False):
        st.caption("옵션 % 계산에만 사용됩니다. 실제 passive_power는 '잔여 전투력'로 계산됩니다.")
        passive_start_atk = st.number_input("시작 passiveAtk", min_value=0.0, value=75.0, step=1.0)
        passive_start_hp = st.number_input("시작 passiveHp", min_value=0.0, value=300.0, step=10.0)
        passive_max_atk = st.number_input("최대 passiveAtk", min_value=0.0, value=900.0, step=10.0)
        passive_max_hp = st.number_input("최대 passiveHp", min_value=0.0, value=3600.0, step=50.0)
        passive_max_stage = st.number_input("최대 적용 스테이지", min_value=1, value=80, step=1)
        passive_step_stage = st.number_input("증가 구간(스테이지)", min_value=1, value=10, step=1)

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
        udf_raw = parse_user_csv(f)
        stage_rows, vinfo = build_stage_rows_for_user(
            df=udf_raw,
            user_label=user_label,
            agency_map=agency_map,
            slotType_to_type_id=slotType_to_type_id,
            item_map=item_map,
            characterAtk=characterAtk,
            characterHp=characterHp,
            passive_start_atk=passive_start_atk,
            passive_start_hp=passive_start_hp,
            passive_max_atk=passive_max_atk,
            passive_max_hp=passive_max_hp,
            passive_max_stage=int(passive_max_stage),
            passive_step_stage=int(passive_step_stage),
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
df_all = df_all[(df_all["stage_lv"] >= 1) & (df_all["stage_lv"] <= int(max_stage))].copy()

# display-friendly strings
df_all["equips"] = df_all["equips_names"].apply(names_join)
df_all["runes"] = df_all["runes_names"].apply(names_join)
df_all["equips_gradeTypes"] = df_all["equips_grades"].apply(names_join)
df_all["runes_gradeTypes"] = df_all["runes_grades"].apply(names_join)
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
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 2])
    with c1:
        stage_min = st.number_input("stage min", min_value=1, max_value=int(max_stage), value=1, step=1, key="overall_stage_min")
    with c2:
        stage_max = st.number_input("stage max", min_value=1, max_value=int(max_stage), value=int(max_stage), step=1, key="overall_stage_max")
    with c3:
        view_mode = st.selectbox("보기", options=["전투력(절대값)", "비중(%)"], index=0, key="overall_view_mode")
    with c4:
        split_options = st.checkbox("옵션 분리(8개)", value=False, key="overall_split_options")
    with c5:
        users_sel = st.multiselect("유저(선택 시 필터)", options=sorted(df_all["user"].unique()), default=[], key="overall_users")

    view = df_all[(df_all["stage_lv"] >= int(stage_min)) & (df_all["stage_lv"] <= int(stage_max))].copy()
    if users_sel:
        view = view[view["user"].isin(users_sel)].copy()

    stage_agg = build_stage_agg(view)

    if view_mode == "전투력(절대값)":
        long_abs = build_long_for_plot(stage_agg, split_options=split_options, include_total=True, mode="abs")
        order = _order_components_by_last_value(long_abs, "label", "stage_lv", "value")
        fig = px.line(
            long_abs,
            x="stage_lv",
            y="value",
            color="label",
            title="스테이지별 평균 전투력 (전체 + 컨텐츠별)",
            category_orders={"label": order} if order else None,
        )
        fig.update_layout(yaxis_title="전투력(평균)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        long_pct = build_long_for_plot(stage_agg, split_options=split_options, include_total=False, mode="pct")
        order = _order_components_by_last_value(long_pct, "label", "stage_lv", "value")
        fig = px.area(
            long_pct,
            x="stage_lv",
            y="value",
            color="label",
            groupnorm="percent",
            title="스테이지별 컨텐츠 전투력 비중(%)",
            category_orders={"label": order} if order else None,
        )
        fig.update_layout(yaxis_title="비중(%)")
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

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        view_mode_u = st.selectbox("보기", options=["비중(%)", "전투력(절대값)"], index=0, key="user_view_mode")
    with c2:
        split_options_u = st.checkbox("옵션 분리(8개)", value=False, key="user_split_options")
    with c3:
        stage_focus = st.selectbox(
            "상세 확인할 스테이지(선택)",
            options=udf["stage_lv"].tolist(),
            index=len(udf) - 1 if len(udf) > 0 else 0,
            key="user_stage_focus",
        )
    with c4:
        st.caption("유저별 그래프는 한 그래프에서 컨텐츠별 전투력/비중을 확인합니다.")

    stage_u = udf[[
        "stage_lv",
        "total_power",
        "passive_power", "agency_power", "character_power",
        "gear_base_power", "slotLv_power", "rune_base_power",
        "gear_option_power", "rune_option_power",
    ]].copy()

    if view_mode_u == "전투력(절대값)":
        long_u = build_long_for_plot(stage_u, split_options=split_options_u, include_total=True, mode="abs")
        order = _order_components_by_last_value(long_u, "label", "stage_lv", "value")
        fig2 = px.line(
            long_u,
            x="stage_lv",
            y="value",
            color="label",
            title=f"{user_sel}: 스테이지별 전투력 (전체 + 컨텐츠별)",
            category_orders={"label": order} if order else None,
        )
        fig2.update_layout(yaxis_title="전투력")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        long_u = build_long_for_plot(stage_u, split_options=split_options_u, include_total=False, mode="pct")
        order = _order_components_by_last_value(long_u, "label", "stage_lv", "value")
        fig2 = px.area(
            long_u,
            x="stage_lv",
            y="value",
            color="label",
            groupnorm="percent",
            title=f"{user_sel}: 스테이지별 컨텐츠 전투력 비중(%)",
            category_orders={"label": order} if order else None,
        )
        fig2.update_layout(yaxis_title="비중(%)")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("선택 스테이지 상세 분해(검증용) — 패시브 제외 나머지 컨텐츠")

    snap = udf[udf["stage_lv"] == int(stage_focus)].iloc[0]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("total_power(로그)", f"{snap['total_power']:.0f}")
    s2.metric("passive_power(잔여)", f"{snap['passive_power']:.0f}")
    s3.metric("non_passive_sum", f"{snap['non_passive_sum']:.0f}")
    s4.metric("agency_lv", str(int(snap["agency_lv"])))

    s5, s6, s7, s8 = st.columns(4)
    s5.metric("agency_power", f"{snap['agency_power']:.0f}")
    s6.metric("character_power", f"{snap['character_power']:.0f}")
    s7.metric("gear_base_power", f"{snap['gear_base_power']:.0f}")
    s8.metric("slotLv_power", f"{snap['slotLv_power']:.0f}")

    s9, s10, s11, s12 = st.columns(4)
    s9.metric("rune_base_power", f"{snap['rune_base_power']:.0f}")
    s10.metric("gear_option_power", f"{snap['gear_option_power']:.0f}")
    s11.metric("rune_option_power", f"{snap['rune_option_power']:.0f}")
    s12.metric("옵션용 passive atk/hp", f"{snap['assumed_passiveAtk_for_option']:.0f} / {snap['assumed_passiveHp_for_option']:.0f}")

    slot_levels_df, equip_df, rune_df = build_breakdown_tables_for_snapshot(
        snapshot_row=snap,
        slotType_to_type_id=slotType_to_type_id,
        item_map=item_map,
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
    # 옵션은 합쳐서 보여주기:
    df_tmp = df_all.copy()
    df_tmp["gear_power"] = df_tmp["gear_base_power"].astype(float) + df_tmp["gear_option_power"].astype(float)
    df_tmp["rune_power"] = df_tmp["rune_base_power"].astype(float) + df_tmp["rune_option_power"].astype(float)

    agg = (
        df_tmp
        .groupby("stage_lv", as_index=False)
        .agg(
            n_users=("user", "nunique"),
            avg_total_power=("total_power", "mean"),
            avg_passive_power=("passive_power", "mean"),
            avg_agency_power=("agency_power", "mean"),
            avg_character_power=("character_power", "mean"),
            avg_gear_power=("gear_power", "mean"),
            avg_rune_power=("rune_power", "mean"),
            avg_slotLv_power=("slotLv_power", "mean"),
            top_gear_grade=("equips_grades", _mode_from_list_series),
            top_rune_grade=("runes_grades", _mode_from_list_series),
        )
        .sort_values("stage_lv")
    )

    # ✅ 표는 반올림 정수
    round_cols = [c for c in agg.columns if c.startswith("avg_")]
    agg_disp = agg.copy()
    for c in round_cols:
        agg_disp[c] = agg_disp[c].round(0).astype(int)

    c1, c2 = st.columns([1, 3])
    with c1:
        metric_a = st.selectbox(
            "평균 지표",
            options=[
                "avg_total_power",
                "avg_passive_power",
                "avg_agency_power",
                "avg_character_power",
                "avg_gear_power",
                "avg_rune_power",
                "avg_slotLv_power",
            ],
            index=0,
            key="avg_metric",
        )
    with c2:
        fig3 = px.line(
            agg_disp,
            x="stage_lv",
            y=metric_a,
            title=f"Stage vs {metric_a} (반올림)",
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(agg_disp, use_container_width=True, height=520)


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
            "passive_clamped_rows",
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
