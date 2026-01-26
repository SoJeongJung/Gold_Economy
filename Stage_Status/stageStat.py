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


@dataclass
class ItemRow:
    id: str
    name: str
    slotType: str
    atkBase: float
    atkInc: float
    hpBase: float
    hpInc: float
    optionAtk: int          # total-atk% (applies to total base atk pool)
    optionHp: int           # total-hp%  (applies to total base hp pool)
    optionAtkBase: int      # self-atk% (applies to the item's own atk)
    optionHpBase: int       # self-hp%  (applies to the item's own hp)


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
) -> Tuple[
    float, float, float, float,                 # equip_power, rune_power, equip_base_power, equip_level_power
    float, float, float, float,                 # equip_atk, equip_hp, equip_base_atk, equip_base_hp
    float, float,                               # rune_atk, rune_hp
    float, float,                               # optionAtk_pct_sum, optionHp_pct_sum
    float, float,                               # optionAtk_power_self, optionHp_power_self
    List[str], List[str],                       # equip_names, rune_names
]:
    """
    option_total(%)은 여기서 '퍼센트 합'만 모으고, 실제 option_total 전투력은
    agency까지 포함한 base_total_atk/hp를 build_stage_rows_for_user에서 계산한다.
    """
    equip_power = 0.0
    equip_base_power = 0.0
    rune_power = 0.0

    equip_atk = 0.0
    equip_hp = 0.0
    equip_base_atk = 0.0
    equip_base_hp = 0.0

    rune_atk = 0.0
    rune_hp = 0.0

    equip_names: List[str] = []
    rune_names: List[str] = []

    optionAtk_pct_sum = 0.0
    optionHp_pct_sum = 0.0

    optionAtk_power_self = 0.0
    optionHp_power_self = 0.0

    for eid in equip_ids:
        it = item_map.get(str(eid))
        if not it:
            equip_names.append(f"(missing:{eid})")
            continue

        equip_names.append(it.name)

        type_id = slotType_to_type_id.get(it.slotType)
        lv = slot_lv_by_type_id.get(int(type_id), 0) if type_id is not None else 0

        atk, hp, p = calc_item_power(it, lv, is_equip=True)
        atk0, hp0, base_p = calc_item_power(it, 0, is_equip=True)

        equip_power += p
        equip_base_power += base_p

        equip_atk += atk
        equip_hp += hp
        equip_base_atk += atk0
        equip_base_hp += hp0

        optionAtk_pct_sum += (it.optionAtk / 100.0)
        optionHp_pct_sum += (it.optionHp / 100.0)

        if it.optionAtkBase:
            optionAtk_power_self += (atk * (it.optionAtkBase / 100.0)) * 4.0
        if it.optionHpBase:
            optionHp_power_self += (hp * (it.optionHpBase / 100.0))

    for rid in rune_ids:
        it = item_map.get(str(rid))
        if not it:
            rune_names.append(f"(missing:{rid})")
            continue

        rune_names.append(it.name)

        atk, hp, p = calc_item_power(it, 0, is_equip=False)
        rune_power += p
        rune_atk += atk
        rune_hp += hp

        optionAtk_pct_sum += (it.optionAtk / 100.0)
        optionHp_pct_sum += (it.optionHp / 100.0)
        # 룬은 optionAtkBase/optionHpBase 없음(있어도 무시)

    equip_level_power = equip_power - equip_base_power

    # 정수화(표시/합산 안정용)
    optionAtk_power_self = float(int(round(optionAtk_power_self)))
    optionHp_power_self = float(int(round(optionHp_power_self)))

    return (
        equip_power,
        rune_power,
        equip_base_power,
        equip_level_power,
        equip_atk,
        equip_hp,
        equip_base_atk,
        equip_base_hp,
        rune_atk,
        rune_hp,
        optionAtk_pct_sum,
        optionHp_pct_sum,
        optionAtk_power_self,
        optionHp_power_self,
        equip_names,
        rune_names,
    )


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


def build_stage_rows_for_user(
    df: pd.DataFrame,
    user_label: str,
    agency_power_map: Dict[int, float],
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

            (
                equip_power,
                rune_power,
                equip_base_power,
                equip_level_power,
                equip_atk,
                equip_hp,
                equip_base_atk,
                equip_base_hp,
                rune_atk,
                rune_hp,
                optionAtk_pct_sum,
                optionHp_pct_sum,
                optionAtk_power_self,
                optionHp_power_self,
                equip_names,
                rune_names,
            ) = sum_equip_rune_power(
                equip_ids=equips,
                rune_ids=runes,
                slot_lv_by_type_id=slot_lv_by_type_id,
                slotType_to_type_id=slotType_to_type_id,
                item_map=item_map,
            )

            agency_power = float(agency_power_map.get(int(agency_lv_current), 0.0))

            # ---- NEW: Agency atk/hp for option-total pool (user-defined rule) ----
            agency_hp = agency_power / 2.0
            agency_atk = agency_power / 4.0

            # ---- NEW: base_total_atk/hp for "옵션 토탈" pool ----
            equip_level_atk = equip_atk - equip_base_atk
            equip_level_hp = equip_hp - equip_base_hp

            base_total_atk = rune_atk + agency_atk + equip_base_atk + equip_level_atk
            base_total_hp = rune_hp + agency_hp + equip_base_hp + equip_level_hp

            optionAtk_power_total = float(int(round((base_total_atk * optionAtk_pct_sum) * 4.0)))
            optionHp_power_total = float(int(round((base_total_hp * optionHp_pct_sum))))

            row = {
                "user": user_label,
                "time": float(r.get("time", 0.0) or 0.0),
                "stage_lv": stage_lv,
                "stage_id": str(r.get("stage_id", "")),
                "total_power": total_power,

                "equip_power": equip_power,
                "rune_power": rune_power,
                "agency_lv": int(agency_lv_current),
                "agency_power": agency_power,

                "equip_base_power": equip_base_power,
                "equip_level_power": equip_level_power,

                "equips_ids": equips,
                "runes_ids": runes,
                "equips_names": equip_names,
                "runes_names": rune_names,

                # 옵션 기여(토탈/셀프)
                "optionAtk_power_total": optionAtk_power_total,
                "optionHp_power_total": optionHp_power_total,
                "optionAtk_power_self": optionAtk_power_self,
                "optionHp_power_self": optionHp_power_self,

                # (디버깅/검증용) atk/hp 분해값도 저장
                "equip_atk": equip_atk,
                "equip_hp": equip_hp,
                "rune_atk": rune_atk,
                "rune_hp": rune_hp,
                "agency_atk": agency_atk,
                "agency_hp": agency_hp,
                "optionAtk_pct_sum": optionAtk_pct_sum,
                "optionHp_pct_sum": optionHp_pct_sum,
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
                "slotType": "",
                "slot_type_id": None,
                "slot_lv": None,
                "atk": 0.0,
                "hp": 0.0,
                "power": 0.0,
                "base_power(slot_lv=0)": 0.0,
                "level_power": 0.0,
                "optionAtk%": 0,
                "optionHp%": 0,
                "optionAtkBase%": 0,
                "optionHpBase%": 0,
            })
            continue

        type_id = slotType_to_type_id.get(it.slotType)
        lv = slot_lv_by_type_id.get(int(type_id), 0) if type_id is not None else 0

        atk, hp, p = calc_item_power(it, lv, is_equip=True)
        _, _, base = calc_item_power(it, 0, is_equip=True)

        equip_rows.append({
            "name": it.name,
            "equip_id": it.id,
            "slotType": it.slotType,
            "slot_type_id": type_id,
            "slot_lv": lv,
            "atk": atk,
            "hp": hp,
            "power": p,
            "base_power(slot_lv=0)": base,
            "level_power": p - base,
            "optionAtk%": it.optionAtk,
            "optionHp%": it.optionHp,
            "optionAtkBase%": it.optionAtkBase,
            "optionHpBase%": it.optionHpBase,
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
                "atk": 0.0,
                "hp": 0.0,
                "power": 0.0,
                "optionAtk%": 0,
                "optionHp%": 0,
            })
            continue

        atk, hp, p = calc_item_power(it, 0, is_equip=False)
        rune_rows.append({
            "name": it.name,
            "rune_id": it.id,
            "slotType": it.slotType,
            "atk": atk,
            "hp": hp,
            "power": p,
            "optionAtk%": it.optionAtk,
            "optionHp%": it.optionHp,
        })

    rune_df = pd.DataFrame(rune_rows)
    return slot_levels_df, equip_df, rune_df


def names_join(xs: Any) -> str:
    if isinstance(xs, list):
        return ", ".join([str(x) for x in xs])
    if xs is None or (isinstance(xs, float) and np.isnan(xs)):
        return ""
    return str(xs)


def build_multi_series_df(
    df_in: pd.DataFrame,
    x_col: str,
    series_map: Dict[str, str],  # label -> column_name
    enabled_labels: List[str],
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    use_labels = [lab for lab in enabled_labels if lab in series_map]
    parts = []
    for lab in use_labels:
        col = series_map[lab]
        if col not in df_in.columns:
            continue
        tmp = df_in[[x_col] + ([group_col] if group_col else []) + [col]].copy()
        tmp = tmp.rename(columns={col: "value"})
        tmp["series"] = lab
        parts.append(tmp)
    if not parts:
        return pd.DataFrame(columns=[x_col, "value", "series"] + ([group_col] if group_col else []))
    return pd.concat(parts, ignore_index=True)


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
        "룩업 CSV 2 (Items) — 컬럼: id, name, slotType, atkBase, atkInc, hpBase, hpInc, optionAtk, optionHp, optionAtkBase, optionHpBase",
        type=["csv"],
        accept_multiple_files=False,
    )

    st.divider()
    st.subheader("slotType 매핑")
    st.caption("Item.slotType 문자열 → 유저 CSV의 slot_type(1~6)로 매핑 (편집 가능)")

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
    st.info("왼쪽에서 Agency 룩업 CSV(lvl, agency_power)를 업로드하세요.")
    st.stop()
if item_lookup_file is None:
    st.info("왼쪽에서 Item 룩업 CSV를 업로드하세요.")
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

# 장비 전투력(옵션 포함)
df_all["equip_power_total"] = (
    df_all["equip_power"]
    + df_all["optionAtk_power_total"]
    + df_all["optionHp_power_total"]
    + df_all["optionAtk_power_self"]
    + df_all["optionHp_power_self"]
)

# calc_sum_power (옵션 포함)
df_all["calc_sum_power"] = df_all["equip_power_total"] + df_all["rune_power"] + df_all["agency_power"]
df_all["gap_total_minus_calc"] = df_all["total_power"] - df_all["calc_sum_power"]


def _gap_label(g: float) -> str:
    gi = int(round(float(g)))
    if gi == 0:
        return "정확"
    if gi > 0:
        return f"부족 {gi}"
    return f"초과 {abs(gi)}"


df_all["gap_label"] = df_all["gap_total_minus_calc"].apply(_gap_label)

# 그 외 옵션 전투력(잔차)
df_all["other_option_power"] = df_all["total_power"] - (
    df_all["equip_power_total"] + df_all["rune_power"] + df_all["agency_power"]
)

# Name-based list strings for tables
df_all["equips"] = df_all["equips_names"].apply(names_join)
df_all["runes"] = df_all["runes_names"].apply(names_join)

# Keep ID strings too (for debugging)
df_all["equips_ids_str"] = df_all["equips_ids"].apply(names_join)
df_all["runes_ids_str"] = df_all["runes_ids"].apply(names_join)


# =========================
# Series definitions (6)
# =========================
SERIES_LABEL_TO_COL = {
    "전체 전투력": "total_power",
    "장비 전투력": "equip_power_total",
    "룬 전투력": "rune_power",
    "장비 레벨 전투력": "equip_level_power",
    "에이전시 레벨 전투력": "agency_power",
    "그 외 옵션 전투력": "other_option_power",
}
SERIES_ORDER = list(SERIES_LABEL_TO_COL.keys())


def render_series_toggles(prefix: str) -> List[str]:
    st.markdown("**그래프 표시 항목(체크로 on/off)**")
    cols = st.columns(6)
    enabled = []
    for i, lab in enumerate(SERIES_ORDER):
        with cols[i]:
            if st.checkbox(lab, value=True, key=f"{prefix}_series_{i}"):
                enabled.append(lab)
    return enabled


# =========================
# UI: Tabs
# =========================
tab_overall, tab_user, tab_avg, tab_validate = st.tabs(["전체(그래프)", "유저별(그래프+검증)", "스테이지 평균", "데이터 오류 검증"])


# -------------------------
# Tab: Overall graphs + full table
# -------------------------
with tab_overall:
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        stage_min = st.number_input("stage min", min_value=1, max_value=int(max_stage), value=1, step=1, key="overall_stage_min")
    with c2:
        stage_max = st.number_input("stage max", min_value=1, max_value=int(max_stage), value=int(max_stage), step=1, key="overall_stage_max")
    with c3:
        users_sel = st.multiselect("유저(선택 시 필터)", options=sorted(df_all["user"].unique()), default=[], key="overall_users")

    enabled_labels = render_series_toggles(prefix="overall")

    view = df_all[(df_all["stage_lv"] >= int(stage_min)) & (df_all["stage_lv"] <= int(stage_max))].copy()
    if users_sel:
        view = view[view["user"].isin(users_sel)].copy()

    long_df = build_multi_series_df(
        df_in=view,
        x_col="stage_lv",
        series_map=SERIES_LABEL_TO_COL,
        enabled_labels=enabled_labels,
        group_col=None,
    )

    fig = px.line(
        long_df.sort_values(["series", "stage_lv"]),
        x="stage_lv",
        y="value",
        color="series",
        markers=False,
        title="전체 – 스테이지별 전투력 구성(선택 항목)",
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
        show_points = st.checkbox("포인트 표시", value=True, key="user_points")
    with c2:
        stage_focus = st.selectbox(
            "상세 확인할 스테이지(선택)",
            options=udf["stage_lv"].tolist(),
            index=len(udf) - 1 if len(udf) > 0 else 0,
            key="user_stage_focus",
        )
    with c3:
        enabled_labels_u = render_series_toggles(prefix="user")

    long_u = build_multi_series_df(
        df_in=udf,
        x_col="stage_lv",
        series_map=SERIES_LABEL_TO_COL,
        enabled_labels=enabled_labels_u,
        group_col=None,
    )

    fig_u = px.line(
        long_u.sort_values(["series", "stage_lv"]),
        x="stage_lv",
        y="value",
        color="series",
        markers=bool(show_points),
        title=f"{user_sel} – 스테이지별 전투력 구성(선택 항목)",
    )
    st.plotly_chart(fig_u, use_container_width=True)

    st.divider()
    st.subheader("선택 스테이지 상세 분해 (장비/룬/슬롯레벨 확인)")

    snap = udf[udf["stage_lv"] == int(stage_focus)].iloc[0]
    slot_levels_df, equip_df, rune_df = build_breakdown_tables_for_snapshot(
        snapshot_row=snap,
        slotType_to_type_id=slotType_to_type_id,
        item_map=item_map,
    )

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("total_power(로그)", f"{snap['total_power']:.0f}")
    s2.metric("equip_power_total(옵션 포함)", f"{snap['equip_power_total']:.0f}")
    s3.metric("rune_power(계산)", f"{snap['rune_power']:.0f}")
    s4.metric("agency_power(룩업)", f"{snap['agency_power']:.0f}")

    s5, s6, s7, s8 = st.columns(4)
    s5.metric("calc_sum_power", f"{snap['calc_sum_power']:.0f}")
    s6.metric("gap_total_minus_calc", f"{snap['gap_total_minus_calc']:.0f}")
    s7.metric("gap_label", snap["gap_label"])
    s8.metric("agency_lv", str(int(snap["agency_lv"])))

    st.caption(
        "옵션 토탈(%)은 (룬+에이전시+장비+장비레벨)의 ATK/HP 풀을 기준으로 계산됩니다. "
        "other_option_power는 위 합산으로 설명되지 않는 잔차입니다."
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
            avg_total_power=("total_power", "mean"),
            avg_equip_power_total=("equip_power_total", "mean"),
            avg_rune_power=("rune_power", "mean"),
            avg_equip_level_power=("equip_level_power", "mean"),
            avg_agency_power=("agency_power", "mean"),
            avg_other_option_power=("other_option_power", "mean"),
            avg_calc_sum_power=("calc_sum_power", "mean"),
            avg_gap=("gap_total_minus_calc", "mean"),
        )
        .sort_values("stage_lv")
    )

    enabled_labels_a = render_series_toggles(prefix="avg")

    SERIES_LABEL_TO_AVG_COL = {
        "전체 전투력": "avg_total_power",
        "장비 전투력": "avg_equip_power_total",
        "룬 전투력": "avg_rune_power",
        "장비 레벨 전투력": "avg_equip_level_power",
        "에이전시 레벨 전투력": "avg_agency_power",
        "그 외 옵션 전투력": "avg_other_option_power",
    }

    long_a = build_multi_series_df(
        df_in=agg,
        x_col="stage_lv",
        series_map=SERIES_LABEL_TO_AVG_COL,
        enabled_labels=enabled_labels_a,
        group_col=None,
    )

    fig_a = px.line(
        long_a.sort_values(["series", "stage_lv"]),
        x="stage_lv",
        y="value",
        color="series",
        markers=False,
        title="스테이지 평균 – 전투력 구성(선택 항목)",
    )
    st.plotly_chart(fig_a, use_container_width=True)

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
