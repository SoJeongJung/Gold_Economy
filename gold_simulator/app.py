import streamlit as st
import pandas as pd
import math
from typing import Dict, Any, Optional, Tuple

# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="경험치·골드 재화 시뮬레이션", layout="wide")
st.title("경험치·골드 재화 시뮬레이션")

# =========================================================
# Utils
# =========================================================
def to_int(x):
    if pd.isna(x):
        return 0
    if isinstance(x, str):
        return int(x.replace(",", "").strip())
    return int(x)

def floor_int(x) -> int:
    return int(math.floor(x))

def pct(part: float, whole: float) -> float:
    return (part / whole * 100.0) if whole > 0 else 0.0

def balance_ratio(balance: int, income_gold: int) -> float:
    if income_gold <= 0:
        return -999.0
    return balance / income_gold

def target_classify(balance: int, income: int) -> str:
    low = float(st.session_state["target_low"])
    high = float(st.session_state["target_high"])
    r = balance_ratio(balance, income)
    if r < low:
        return "과부족"
    if low <= r <= high:
        return "목표"
    return "여유"


def status_badge(label: str) -> str:
    if label == "과부족":
        return "🟥 과부족"
    if label == "목표":
        return "🟩 목표"
    return "🟨 여유"

def period_to_days(unit: str, value: int) -> int:
    if unit == "일":
        return int(value)
    if unit == "월":
        return int(value) * 30
    return int(value) * 365

# =========================================================
# Defaults / Cohorts
# =========================================================
DEFAULTS = {
    "period_unit": "일",
    "period_value": 30,
    "view_day": 30,

    # 자동 보정 목표 밴드(직접 입력)
    "target_low": -0.25,
    "target_high": -0.20,

    # 골드 슬롯 파라미터(조정 가능) - 편집 토글
    "slot_params_edit_on": False,
    "base_slot_multiplier": 1.866,
    "bet_none_multiplier": 2.10,        # 월간 미구매 배팅 1배
    "bet_monthly_multiplier": 1.85,     # 월간 구매 배팅 1배(배팅 없음)
    "bet_2x_multiplier": 2.708,
    "bet_4x_multiplier": 4.80,
    "bet_2x_cost": 40,
    "bet_4x_cost": 120,

    # 장비
    "use_gear_table": True,
    "gear_factor": 1.0,
    "gear_offset": 0,

    # 추가 XP 정책(수동값 적용)
    # - 소탕 카드 구매 유저에게만 "무조건" 적용 (옵션 제거)
    "xp_boost_N": 2.00,
}


COHORTS = ["무과금", "소과금", "중과금", "핵과금"]

# 코호트별 기본 정책
# 변경점: "목표 스테이지(=240) 도달 기간"을 코호트별로 입력 (days_to_max_stage)
COHORT_DEFAULTS = {
    "무과금": {
        "start_stage": 1,
        "days_to_max_stage": 1460,   # 예: 4년
        "alpha": 0.60,

        "minutes_per_energy": 12,
        "free_5_energy": True,
        "buy_10_energy": 0,
        "buy_20_energy": 0,
        "main_play": 20,

        "sweep_card_on": False,

        "epic_card_on": False,
        "gold_slot_monthly_on": False,
        "bet_choice": "월간 미구매(고정)",
        "chips_used": 0,
    },
    "소과금": {
        "start_stage": 1,
        "days_to_max_stage": 900,    # 예: 2.5년
        "alpha": 0.55,

        "minutes_per_energy": 12,
        "free_5_energy": True,
        "buy_10_energy": 1,
        "buy_20_energy": 1,
        "main_play": 5,

        "sweep_card_on": True,

        "epic_card_on": True,
        "gold_slot_monthly_on": False,
        "bet_choice": "월간 미구매(고정)",
        "chips_used": 0,
    },
    "중과금": {
        "start_stage": 1,
        "days_to_max_stage": 540,    # 예: 1.5년
        "alpha": 0.50,

        "minutes_per_energy": 12,
        "free_5_energy": True,
        "buy_10_energy": 1,
        "buy_20_energy": 5,
        "main_play": 5,

        "sweep_card_on": True,

        "epic_card_on": True,
        "gold_slot_monthly_on": True,
        "bet_choice": "배팅 없음",
        "chips_used": 10,
    },
    "핵과금": {
        "start_stage": 1,
        "days_to_max_stage": 365,    # 예: 1년
        "alpha": 0.45,

        "minutes_per_energy": 12,
        "free_5_energy": True,
        "buy_10_energy": 1,
        "buy_20_energy": 20,
        "main_play": 5,

        "sweep_card_on": True,

        "epic_card_on": True,
        "gold_slot_monthly_on": True,
        "bet_choice": "4배 배팅",
        "chips_used": 20,
    },
}

def k(cohort: str, name: str) -> str:
    return f"{cohort}__{name}"

def apply_cohort_defaults():
    for cohort in COHORTS:
        for name, val in COHORT_DEFAULTS[cohort].items():
            key = k(cohort, name)
            if key not in st.session_state:
                st.session_state[key] = val
        # 코호트별 자동보정 저장소
        if k(cohort, "auto_xp_mult_reco") not in st.session_state:
            st.session_state[k(cohort, "auto_xp_mult_reco")] = 1.0
        if k(cohort, "auto_xp_mult_apply") not in st.session_state:
            st.session_state[k(cohort, "auto_xp_mult_apply")] = 1.0
        if k(cohort, "auto_reco_info") not in st.session_state:
            st.session_state[k(cohort, "auto_reco_info")] = None

def reset_all():
    for name, val in DEFAULTS.items():
        st.session_state[name] = val
    for cohort in COHORTS:
        for name, val in COHORT_DEFAULTS[cohort].items():
            st.session_state[k(cohort, name)] = val
        st.session_state[k(cohort, "auto_xp_mult_reco")] = 1.0
        st.session_state[k(cohort, "auto_xp_mult_apply")] = 1.0
        st.session_state[k(cohort, "auto_reco_info")] = None

for name, val in DEFAULTS.items():
    if name not in st.session_state:
        st.session_state[name] = val
apply_cohort_defaults()

# =========================================================
# Passive GOLD rate (HARD-CODED)
# =========================================================
def passive_gold_rate_from_level(level: int) -> float:
    if level <= 0:
        return 0.0
    if level >= 110:
        return 0.25
    step = ((level - 1) // 5) + 1
    return 0.01 * step

# =========================================================
# Load CSV
# =========================================================
import io

# =========================================================
# Upload CSV (NEW)
# =========================================================
st.sidebar.header("CSV 업로드")
st.sidebar.caption("필수 4개 CSV를 업로드하면 시뮬레이션이 실행됩니다.")

up_stage = st.sidebar.file_uploader("stage_economy.csv 업로드", type=["csv"], key="up_stage")
up_passive = st.sidebar.file_uploader("passive_cost.csv 업로드", type=["csv"], key="up_passive")
up_level = st.sidebar.file_uploader("account_level.csv 업로드", type=["csv"], key="up_level")
up_gear = st.sidebar.file_uploader("gear_level.csv 업로드", type=["csv"], key="up_gear")

@st.cache_data
def load_data_from_uploads(stage_bytes: bytes, passive_bytes: bytes, level_bytes: bytes, gear_bytes: bytes):
    stage_df = pd.read_csv(io.BytesIO(stage_bytes))
    passive_df = pd.read_csv(io.BytesIO(passive_bytes))
    level_df = pd.read_csv(io.BytesIO(level_bytes))
    gear_df = pd.read_csv(io.BytesIO(gear_bytes))

    # ===== 기존 load_data()의 검증/전처리 로직을 그대로 유지 =====
    required_stage_cols = ["stage", "xp", "gold_stage_play", "gold_shop_free", "gold_dungeon"]
    missing = [c for c in required_stage_cols if c not in stage_df.columns]
    if missing:
        raise ValueError(f"stage_economy.csv 필수 컬럼 누락: {missing}")

    stage_df["stage"] = stage_df["stage"].apply(to_int)
    stage_df["xp"] = stage_df["xp"].apply(to_int)
    stage_df["gold_stage_play"] = stage_df["gold_stage_play"].apply(to_int)
    stage_df["gold_shop_free"] = stage_df["gold_shop_free"].apply(to_int)
    stage_df["gold_dungeon"] = stage_df["gold_dungeon"].apply(to_int)

    passive_df["passive_draw_count"] = passive_df["passive_draw_count"].apply(to_int)
    passive_df["required_account_level"] = passive_df["required_account_level"].apply(to_int)
    passive_df["gold_cost"] = passive_df["gold_cost"].apply(to_int)
    passive_df = passive_df.sort_values("passive_draw_count").reset_index(drop=True)

    level_df["userLevel"] = level_df["userLevel"].apply(to_int)
    level_df["minXp"] = level_df["minXp"].apply(to_int)
    level_df["needXp"] = level_df["needXp"].apply(to_int)
    level_df = level_df.sort_values("userLevel").reset_index(drop=True)

    gear_df["gear_level"] = gear_df["gear_level"].apply(to_int)
    gear_df["need_gold"] = gear_df["need_gold"].apply(to_int)
    gear_df = gear_df.sort_values("gear_level").reset_index(drop=True)

    if stage_df["stage"].min() < 1:
        raise ValueError("stage_economy.csv: stage는 1 이상이어야 합니다.")
    if stage_df["stage"].duplicated().any():
        raise ValueError("stage_economy.csv: stage가 중복됩니다. (stage는 유일해야 함)")

    return stage_df, passive_df, level_df, gear_df

# 업로드가 모두 되어야 실행
if not (up_stage and up_passive and up_level and up_gear):
    st.warning("왼쪽 사이드바에서 4개 CSV를 모두 업로드해주세요: stage_economy / passive_cost / account_level / gear_level")
    st.stop()

try:
    stage_df, passive_df, level_df, gear_df = load_data_from_uploads(
        up_stage.getvalue(),
        up_passive.getvalue(),
        up_level.getvalue(),
        up_gear.getvalue(),
    )
except Exception as e:
    st.error(f"데이터 로딩 오류: {e}")
    st.stop()

# 기존 후속 로직은 그대로
stage_map = stage_df.set_index("stage").to_dict(orient="index")
MAX_STAGE = int(stage_df["stage"].max())
MIN_STAGE = int(stage_df["stage"].min())



# =========================================================
# Account level calc
# =========================================================
def calc_account_level(total_xp: int) -> Tuple[int, int, int, float]:
    eligible = level_df[level_df["minXp"] <= total_xp]
    row = eligible.iloc[-1] if not eligible.empty else level_df.iloc[0]
    lvl = int(row["userLevel"])
    cur_min = int(row["minXp"])
    need = int(row["needXp"])
    xp_in_level = max(total_xp - cur_min, 0)
    progress = (xp_in_level / need) if need > 0 else 1.0
    progress = min(max(progress, 0.0), 1.0)
    return lvl, xp_in_level, need, progress

# =========================================================
# Gear prefix
# =========================================================
gear_level_max = int(gear_df["gear_level"].max()) if not gear_df.empty else 0

def build_gear_prefix():
    cost_by_lvl = {int(r["gear_level"]): int(r["need_gold"]) for _, r in gear_df.iterrows()}
    prefix = [0] * (gear_level_max + 1)
    for lvl in range(1, gear_level_max + 1):
        prefix[lvl] = prefix[lvl - 1] + cost_by_lvl.get(lvl, 0)
    return prefix

gear_prefix = build_gear_prefix() if gear_level_max > 0 else [0]

def gear_total_cost_for_level(lvl: int) -> int:
    if gear_level_max <= 0:
        return 0
    lvl = max(0, min(lvl, gear_level_max))
    return gear_prefix[lvl]

# =========================================================
# Passive scenarios (A / B 변경)
# =========================================================
def unlocked_df_at_or_below(level: int) -> pd.DataFrame:
    if level <= 0:
        return passive_df.iloc[0:0]
    return passive_df[passive_df["required_account_level"] <= level].sort_values("passive_draw_count")

def spend_draws_at_level_all(level: int) -> Tuple[int, int]:
    u = unlocked_df_at_or_below(level)
    spend = int(u["gold_cost"].sum()) if not u.empty else 0
    draws = int(len(u))
    return spend, draws

def spend_draws_at_level_partial(level: int, fraction: float) -> Tuple[int, int]:
    """
    'level까지 해금된 패시브' 중, 앞에서부터 일부만 뽑는다고 가정.
    fraction: 0~1
    """
    u = unlocked_df_at_or_below(level)
    if u.empty:
        return 0, 0
    fraction = max(0.0, min(1.0, float(fraction)))
    n_total = int(len(u))
    n = int(math.floor(n_total * fraction))
    n = max(0, min(n, n_total))
    up = u.iloc[:n]
    spend = int(up["gold_cost"].sum()) if not up.empty else 0
    return spend, n

def scenario_A(final_level: int) -> Tuple[int, int, int]:
    # 기존 A 유지: 최종 레벨이 속한 5레벨 구간의 베이스 레벨까지 전부 뽑음
    base = (final_level // 5) * 5
    spend, draws = spend_draws_at_level_all(base)
    return base, spend, draws

def scenario_B_new(final_level: int) -> Dict[str, Any]:
    """
    요구사항 반영:
    - final=44 -> 40까지 해금된 패시브 '전부'
    - final=45 -> 45까지 해금된 패시브 '1~2회' 수준(아주 일부)
    - final=46 -> 45까지 '2~4회' 수준(조금 더)
    - final=49 -> 45까지 '거의'
    - final=50 -> 45까지 '전부'

    모델링(결정론):
    - completed_cap = ((L-1)//5)*5  (이미 '전부' 완료된 구간)
      예) 44 -> 40, 45~49 -> 40, 50 -> 45
    - inprogress_cap = completed_cap + 5 (현재 진행 중인 해금 구간의 상한)
      예) 45~49 -> 45, 50~54 -> 50
    - L이 inprogress_cap에 진입(=45)하면, inprogress_cap까지 일부 뽑기 시작
      offset = L - inprogress_cap (0~4)
      fraction_map = [0.10, 0.20, 0.40, 0.70, 0.95]  # 45~49
      (정확한 “1~2회” 등을 일관되게 재현하려면 passive 테이블의 draws 수에 따라 달라지므로,
       여기서는 '부분 구매 정도'를 분수로 모델링)
    - L이 다음 5배수(=50)에 도달하면, inprogress_cap(=45)까지는 '전부' 완료로 간주
    """
    L = int(final_level)
    completed_cap = ((L - 1) // 5) * 5  # 50 -> 45
    inprogress_cap = completed_cap + 5  # 50 -> 50, 45 -> 45

    # completed_cap까지는 전부
    completed_spend, completed_draws = spend_draws_at_level_all(completed_cap)

    # inprogress 부분(현재 5레벨 구간)
    partial_cap = inprogress_cap
    partial_fraction = 0.0
    partial_spend = 0
    partial_draws = 0

    if L >= inprogress_cap and (L % 5 != 0):  # 45~49 구간에서만 부분 구매가 존재 (50은 완료 처리로 넘어감)
        offset = L - inprogress_cap  # 0..4
        fraction_map = [0.10, 0.20, 0.40, 0.70, 0.95]
        partial_fraction = fraction_map[max(0, min(4, offset))]
        partial_spend, partial_draws = spend_draws_at_level_partial(partial_cap, partial_fraction)

    # 최종 B는 "completed_cap 전부 + (있다면) inprogress_cap 부분"
    total_spend = completed_spend + partial_spend
    total_draws = completed_draws + partial_draws

    return {
        "completed_cap": completed_cap,
        "partial_cap": partial_cap,
        "partial_fraction": partial_fraction,
        "passive_spend": int(total_spend),
        "draws": int(total_draws),
    }

# =========================================================
# Sidebar UI
# =========================================================
st.sidebar.header("설정")
if st.sidebar.button("전체 기본값 리셋", use_container_width=True):
    reset_all()
    st.rerun()

st.sidebar.divider()

st.sidebar.subheader("시뮬레이션 기간")
period_unit = st.sidebar.selectbox("기간 단위", ["일", "월", "년"], key="period_unit")
period_value = st.sidebar.number_input("기간 값(정수)", min_value=1, max_value=5000, step=1, key="period_value")
simulation_days = period_to_days(period_unit, int(period_value))
st.sidebar.caption(f"내부 계산 기준: {simulation_days}일")

st.sidebar.subheader("보기(현재 시점)")
view_day = st.sidebar.slider(
    "현재 시점(일차)",
    1, max(1, simulation_days),
    value=min(int(st.session_state.get("view_day", simulation_days)), simulation_days),
    key="view_day",
)
st.sidebar.caption("이 슬라이더를 늘리면 각 코호트의 현재 시점 결과가 자동 갱신됩니다.")

st.sidebar.divider()

st.sidebar.subheader("공통 옵션")
use_gear_table = st.sidebar.checkbox("장비 강화 소비를 gear_level.csv로 계산", key="use_gear_table")
gear_factor = st.sidebar.slider("장비레벨 = 계정레벨 × 계수", 0.5, 2.0, step=0.05, key="gear_factor")
gear_offset = st.sidebar.number_input("장비레벨 오프셋(+)", min_value=-50, max_value=200, step=1, key="gear_offset")

st.sidebar.subheader("추가 XP 지급(수동값 적용)")
st.sidebar.caption("수동값 적용 탭에서만 사용. 소탕 카드 구매 유저에게만 적용됩니다(고정 정책).")
xp_boost_N = st.sidebar.number_input("추가 XP 배율(N배)", min_value=1.0, max_value=20.0, step=0.01, key="xp_boost_N")


st.sidebar.subheader("자동 보정 목표 밴드(ratio)")
st.sidebar.caption("ratio = 잔액 / 누적골드(수급). 예: -0.25 = -25%")
st.sidebar.number_input("목표 하한(LOW)", min_value=-0.99, max_value=0.0, step=0.01, key="target_low")
st.sidebar.number_input("목표 상한(HIGH)", min_value=-0.99, max_value=0.0, step=0.01, key="target_high")

# 안전장치: low가 high보다 크면 스왑
if float(st.session_state["target_low"]) > float(st.session_state["target_high"]):
    st.session_state["target_low"], st.session_state["target_high"] = st.session_state["target_high"], st.session_state["target_low"]


st.sidebar.divider()
st.sidebar.subheader("골드 슬롯 파라미터(조정 가능)")
slot_params_edit_on = st.sidebar.checkbox("골드 슬롯 배율/비용 수정하기", key="slot_params_edit_on")
disabled_slot = not bool(st.session_state["slot_params_edit_on"])

base_slot_multiplier = st.sidebar.number_input(
    "base_slot_multiplier", min_value=0.0, value=float(st.session_state["base_slot_multiplier"]),
    step=0.01, key="base_slot_multiplier", disabled=disabled_slot
)
bet_none_multiplier = st.sidebar.number_input(
    "bet_none_multiplier(미구매 1배)", min_value=0.0, value=float(st.session_state["bet_none_multiplier"]),
    step=0.01, key="bet_none_multiplier", disabled=disabled_slot
)
bet_monthly_multiplier = st.sidebar.number_input(
    "bet_monthly_multiplier(구매 1배)", min_value=0.0, value=float(st.session_state["bet_monthly_multiplier"]),
    step=0.01, key="bet_monthly_multiplier", disabled=disabled_slot
)
bet_2x_multiplier = st.sidebar.number_input(
    "bet_2x_multiplier", min_value=0.0, value=float(st.session_state["bet_2x_multiplier"]),
    step=0.01, key="bet_2x_multiplier", disabled=disabled_slot
)
bet_4x_multiplier = st.sidebar.number_input(
    "bet_4x_multiplier", min_value=0.0, value=float(st.session_state["bet_4x_multiplier"]),
    step=0.01, key="bet_4x_multiplier", disabled=disabled_slot
)
bet_2x_cost = st.sidebar.number_input(
    "bet_2x_cost(다이아/스핀)", min_value=0, value=int(st.session_state["bet_2x_cost"]),
    step=1, key="bet_2x_cost", disabled=disabled_slot
)
bet_4x_cost = st.sidebar.number_input(
    "bet_4x_cost(다이아/스핀)", min_value=0, value=int(st.session_state["bet_4x_cost"]),
    step=1, key="bet_4x_cost", disabled=disabled_slot
)


st.sidebar.divider()
st.sidebar.subheader("코호트별 정책(무/소/중/핵)")
st.sidebar.caption(f"스테이지 최대값은 stage_economy 기준 **{MAX_STAGE}** 입니다. 코호트별로 '{MAX_STAGE} 도달 기간'과 alpha를 조정합니다.")

# =========================================================
# Core helpers
# =========================================================
AUTO_SWEEP_COUNT = 1
energy_per_play = 5

def calc_energy_and_play(minutes_per_energy_: int, free_5_energy_: bool, buy_10_energy_: int, buy_20_energy_: int) -> Tuple[int, int, int]:
    charged_energy = floor_int(1440 / int(minutes_per_energy_))
    shop_energy = 5 if free_5_energy_ else 0
    energy_from_10 = int(buy_10_energy_) * 10
    energy_from_20 = int(buy_20_energy_) * 20
    total_energy = charged_energy + shop_energy + energy_from_10 + energy_from_20
    daily_max_play = floor_int(total_energy / energy_per_play)

    diamond_energy_10 = int(buy_10_energy_) * 50
    diamond_energy_20 = 100 * int(buy_20_energy_) * (int(buy_20_energy_) + 1) // 2
    diamond_energy_daily = diamond_energy_10 + diamond_energy_20
    return daily_max_play, charged_energy, diamond_energy_daily

def slot_params(monthly_on: bool, bet_choice: str) -> Tuple[float, int]:
    if monthly_on:
        if bet_choice == "배팅 없음":
            return float(bet_monthly_multiplier), 0
        if bet_choice == "2배 배팅":
            return float(bet_2x_multiplier), int(bet_2x_cost)
        return float(bet_4x_multiplier), int(bet_4x_cost)
    return float(bet_none_multiplier), 0

def stage_at_day_to_max(day_1based: int, start_stage: int, days_to_max_stage: int, alpha: float) -> int:
    """
    요구사항 반영:
    - 스테이지는 "코호트별 목표 스테이지"에서 멈추지 않고,
      MAX_STAGE까지 비선형으로 계속 진행(도달 기간을 코호트별로 입력).
    - days_to_max_stage 이후에는 MAX_STAGE 고정(상한).
    """
    dmax = max(1, int(days_to_max_stage))
    if dmax == 1:
        s = MAX_STAGE
    else:
        t = (day_1based - 1) / (dmax - 1)
        t = min(max(t, 0.0), 1.0)  # 0~1 clamp
        prog = t ** float(alpha)
        s = start_stage + (MAX_STAGE - start_stage) * prog
    s = floor_int(s)
    s = max(MIN_STAGE, min(s, MAX_STAGE))
    return s

def manual_xp_multiplier(sweep_card_on: bool) -> float:
    return float(st.session_state["xp_boost_N"]) if sweep_card_on else 1.0


# =========================================================
# Simulation per cohort
# =========================================================
def run_simulation_for_cohort(cohort: str, auto_xp_mult: float, use_manual: bool) -> Optional[Dict[str, Any]]:
    start_stage = int(st.session_state[k(cohort, "start_stage")])
    days_to_max_stage = int(st.session_state[k(cohort, "days_to_max_stage")])
    alpha = float(st.session_state[k(cohort, "alpha")])

    minutes_per_energy_ = int(st.session_state[k(cohort, "minutes_per_energy")])
    free_5_energy_ = bool(st.session_state[k(cohort, "free_5_energy")])
    buy_10_energy_ = int(st.session_state[k(cohort, "buy_10_energy")])
    buy_20_energy_ = int(st.session_state[k(cohort, "buy_20_energy")])

    sweep_card_on = bool(st.session_state[k(cohort, "sweep_card_on")])

    epic_card_on = bool(st.session_state[k(cohort, "epic_card_on")])
    monthly_on = bool(st.session_state[k(cohort, "gold_slot_monthly_on")])
    bet_choice_ = str(st.session_state[k(cohort, "bet_choice")])
    chips_used_ = int(st.session_state[k(cohort, "chips_used")])

    daily_max_play, charged_energy, diamond_energy_daily = calc_energy_and_play(
        minutes_per_energy_, free_5_energy_, buy_10_energy_, buy_20_energy_
    )
    main_play = int(st.session_state[k(cohort, "main_play")])
    main_play = max(0, min(main_play, daily_max_play))
    quick_sweep = daily_max_play - main_play

    if monthly_on:
        free_spins = 2 + (1 if epic_card_on else 0) + 3
        paid_spins = max(0, min(chips_used_, 20))
        total_spins = free_spins + paid_spins
    else:
        free_spins = 2 + (1 if epic_card_on else 0)
        paid_spins = 0
        total_spins = free_spins
        bet_choice_ = "월간 미구매(고정)"

    bet_mult, bet_cost_per_spin = slot_params(monthly_on, bet_choice_)
    diamond_slot_daily = (paid_spins * 20) + (bet_cost_per_spin * total_spins)
    diamond_daily = int(diamond_energy_daily + diamond_slot_daily)

    records = []
    cum_xp = 0
    cum_dia = 0

    for day in range(1, int(simulation_days) + 1):
        stg = stage_at_day_to_max(day, start_stage, days_to_max_stage, alpha)
        econ = stage_map.get(stg)
        if econ is None:
            continue

        xp = int(econ["xp"])
        gold_stage_play = int(econ["gold_stage_play"])
        gold_shop_free = int(econ["gold_shop_free"])
        gold_dungeon = int(econ["gold_dungeon"])

        # [NEW] 소탕 카드 미구매 유저 페널티: 스테이지별 XP/Gold 30% 감소
        if not sweep_card_on:
             xp = floor_int(xp * 0.70)
             gold_stage_play = floor_int(gold_stage_play * 0.70)
             gold_dungeon = floor_int(gold_dungeon * 0.70)


        # XP base
        main_xp_base = main_play * xp
        quick_xp_base = quick_sweep * xp
        auto_xp_base = AUTO_SWEEP_COUNT * xp * 2


        m_manual = manual_xp_multiplier(sweep_card_on) if use_manual else 1.0
        xp_mult = float(auto_xp_mult) * m_manual

        main_xp = floor_int(main_xp_base * xp_mult)
        quick_xp = floor_int(quick_xp_base * xp_mult)
        auto_xp = floor_int(auto_xp_base * xp_mult)

        daily_xp = main_xp + quick_xp + auto_xp
        cum_xp += daily_xp

        acc_lvl, _, _, _ = calc_account_level(cum_xp)
        passive_gold_rate = passive_gold_rate_from_level(acc_lvl)

        # GOLD base
        main_gold_base = main_play * gold_stage_play
        quick_gold_base = quick_sweep * gold_stage_play
        auto_gold_base = AUTO_SWEEP_COUNT * gold_stage_play * 2
        shop_gold_base = gold_shop_free

        slot_gold_per_spin = gold_dungeon * float(base_slot_multiplier) * float(bet_mult)
        slot_gold_base = total_spins * slot_gold_per_spin

        # Passive 적용: 메인/빠른/자동만 O, 상점무료/슬롯 X
        main_gold = floor_int(main_gold_base * (1 + passive_gold_rate))
        quick_gold = floor_int(quick_gold_base * (1 + passive_gold_rate))
        auto_gold = floor_int(auto_gold_base * (1 + passive_gold_rate))
        shop_gold = floor_int(shop_gold_base)
        slot_gold = floor_int(slot_gold_base)

        daily_gold_total = main_gold + quick_gold + auto_gold + shop_gold + slot_gold

        cum_dia += diamond_daily
        cum_gold = (records[-1]["cum_gold"] if records else 0) + daily_gold_total

        gear_lvl = floor_int(acc_lvl * float(gear_factor) + int(gear_offset))
        gear_lvl = max(0, min(gear_lvl, gear_level_max)) if gear_level_max > 0 else 0

        records.append({
            "day": day,
            "stage": stg,
            "daily_xp": daily_xp,
            "cum_xp": cum_xp,
            "xp_main": main_xp,
            "xp_quick": quick_xp,
            "xp_auto": auto_xp,
            "account_level": acc_lvl,
            "passive_gold_rate": passive_gold_rate,
            "daily_gold": daily_gold_total,
            "cum_gold": cum_gold,
            "gold_main": main_gold,
            "gold_quick": quick_gold,
            "gold_auto": auto_gold,
            "gold_shop": shop_gold,
            "gold_slot": slot_gold,
            "daily_diamond": diamond_daily,
            "cum_diamond": cum_dia,
            "gear_level": gear_lvl,

            "daily_max_play": daily_max_play,
            "main_play": main_play,
            "quick_sweep": quick_sweep,
            "charged_energy": charged_energy,
            "free_spins": free_spins,
            "paid_spins": paid_spins,
            "total_spins": total_spins,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return None

    df_view = df[df["day"] <= int(view_day)]
    if df_view.empty:
        return None

    last = df_view.iloc[-1]
    final_xp = int(last["cum_xp"])
    final_gold = int(last["cum_gold"])
    final_dia = int(last["cum_diamond"])
    final_stage = int(last["stage"])

    final_level, xp_in_level, need_xp, lvl_prog = calc_account_level(final_xp)
    final_gear_level = int(last["gear_level"])
    gear_spend = gear_total_cost_for_level(final_gear_level) if (use_gear_table and gear_level_max > 0) else 0

    # 소비 시나리오
    A_base, A_passive_spend, A_draws = scenario_A(final_level)
    B = scenario_B_new(final_level)

    A_total_spend = int(A_passive_spend + gear_spend)
    B_total_spend = int(B["passive_spend"] + gear_spend)

    A_balance = int(final_gold - A_total_spend)
    B_balance = int(final_gold - B_total_spend)

    # breakdown (현재 시점까지)
    xp_breakdown = pd.DataFrame({
        "획득처": ["메인 플레이", "빠른 소탕", "자동 소탕(1회 고정)"],
        "누적(원값)": [
            int(df_view["xp_main"].sum()),
            int(df_view["xp_quick"].sum()),
            int(df_view["xp_auto"].sum()),
        ],
    })
    xp_breakdown["비율"] = xp_breakdown["누적(원값)"].apply(lambda v: f"{pct(v, final_xp):.1f}%")
    xp_breakdown["누적 XP"] = xp_breakdown["누적(원값)"].map(lambda v: f"{v:,}")
    xp_breakdown = xp_breakdown[["획득처", "누적 XP", "비율"]]

    gold_breakdown = pd.DataFrame({
        "획득처": ["메인 플레이", "빠른 소탕", "자동 소탕(1회 고정)", "상점 무료", "골드 슬롯"],
        "누적(원값)": [
            int(df_view["gold_main"].sum()),
            int(df_view["gold_quick"].sum()),
            int(df_view["gold_auto"].sum()),
            int(df_view["gold_shop"].sum()),
            int(df_view["gold_slot"].sum()),
        ],
    })
    gold_breakdown["비율"] = gold_breakdown["누적(원값)"].apply(lambda v: f"{pct(v, final_gold):.1f}%")
    gold_breakdown["누적 Gold"] = gold_breakdown["누적(원값)"].map(lambda v: f"{v:,}")
    gold_breakdown = gold_breakdown[["획득처", "누적 Gold", "비율"]]

    # 더 이해하기 쉬운 점검 메시지(요구사항 반영)
    xp_sum = int(df_view["xp_main"].sum() + df_view["xp_quick"].sum() + df_view["xp_auto"].sum())
    gold_sum = int(df_view["gold_main"].sum() + df_view["gold_quick"].sum() + df_view["gold_auto"].sum() + df_view["gold_shop"].sum() + df_view["gold_slot"].sum())
    checks = {
        "xp_ok": (xp_sum == final_xp),
        "gold_ok": (gold_sum == final_gold),
        "play_ok": (int(last["main_play"]) + int(last["quick_sweep"]) == int(last["daily_max_play"])),
    }

    ops = {
        "daily_max_play": int(last["daily_max_play"]),
        "main_play": int(last["main_play"]),
        "quick_sweep": int(last["quick_sweep"]),
        "total_spins": int(last["total_spins"]),
        "free_spins": int(last["free_spins"]),
        "paid_spins": int(last["paid_spins"]),
        "daily_diamond": int(last["daily_diamond"]),
        "days_to_max_stage": int(days_to_max_stage),
        "alpha": float(alpha),
        "start_stage": int(start_stage),
    }

    return {
        "df_full": df,
        "df_view": df_view,
        "final": {
            "xp": final_xp,
            "gold": final_gold,
            "diamond": final_dia,
            "stage": final_stage,
            "level": final_level,
            "xp_in_level": xp_in_level,
            "need_xp": need_xp,
            "lvl_prog": lvl_prog,
            "gear_level": final_gear_level,
            "gear_spend": int(gear_spend),
        },
        "A": {
            "base": int(A_base),
            "passive_spend": int(A_passive_spend),
            "draws": int(A_draws),
            "total_spend": int(A_total_spend),
            "balance": int(A_balance),
        },
        "B": {
            **B,
            "total_spend": int(B_total_spend),
            "balance": int(B_balance),
        },
        "xp_breakdown": xp_breakdown,
        "gold_breakdown": gold_breakdown,
        "checks": checks,
        "ops": ops,
    }

# =========================================================
# Auto recommendation per cohort (목표 밴드 변경 반영)
# - 목표: ratio(잔액/누적골드)가 [-25%, -20%] 근처
# - 코호트 목표 시나리오: 기본 B, 핵과금만 A (기존 유지)
# =========================================================
def recommend_auto_multiplier_for_cohort(cohort: str):
    scenario = "A" if cohort == "핵과금" else "B"

    low = float(st.session_state["target_low"])
    high = float(st.session_state["target_high"])

    candidates = []
    for i in range(80):
        t = i / 79
        mult = 10 ** (math.log10(0.25) * (1 - t) + math.log10(20.0) * t)
        candidates.append(mult)

    best_in_band = None
    best_near = None

    for m in candidates:
        res = run_simulation_for_cohort(cohort, m, use_manual=False)
        if res is None:
            continue

        income = res["final"]["gold"]
        bal = res[scenario]["balance"]
        r = balance_ratio(bal, income)

        if low <= r <= high:
            # 밴드 내: 상한(high)에 가장 근접(덜 부족한 쪽으로 여유 최소화)
            score = abs(high - r)
            if best_in_band is None or score < best_in_band["score"]:
                best_in_band = {"m": m, "r": r, "bal": bal, "income": income, "lvl": res["final"]["level"], "score": score}
        else:
            dist = (low - r) if (r < low) else (r - high)
            if best_near is None or dist < best_near["dist"]:
                best_near = {"m": m, "r": r, "bal": bal, "income": income, "lvl": res["final"]["level"], "dist": dist}

    chosen = best_in_band if best_in_band is not None else best_near
    return scenario, chosen


# =========================================================
# Cohort UI (Expander)
# =========================================================
for cohort in COHORTS:
    with st.sidebar.expander(f"{cohort} 설정", expanded=(cohort == "무과금")):
        st.markdown("**스테이지 진행(코호트별)**")
        st.slider("시작 스테이지", MIN_STAGE, MAX_STAGE, key=k(cohort, "start_stage"))
        st.number_input(
            f"{MAX_STAGE} 스테이지 도달 기간(일)",
            min_value=1, max_value=5000, step=1,
            key=k(cohort, "days_to_max_stage"),
        )
        st.slider("비선형 감속(alpha)", 0.20, 1.00, step=0.05, key=k(cohort, "alpha"))
        st.caption("해석: 입력한 도달 기간 동안 MAX_STAGE까지 비선형으로 증가(이후 상한 고정)")

        st.divider()
        st.markdown("**플레이(에너지 기반, 코호트별)**")
        st.number_input("에너지 1개 충전 시간(분)", min_value=1, step=1, key=k(cohort, "minutes_per_energy"))
        st.checkbox("상점 무료 5에너지 받기", key=k(cohort, "free_5_energy"))
        st.number_input("10에너지 구매(50다이아, 1회)", min_value=0, max_value=1, step=1, key=k(cohort, "buy_10_energy"))
        st.number_input("20에너지 구매(최대 20회, 100→200→...)", min_value=0, max_value=20, step=1, key=k(cohort, "buy_20_energy"))

        dm, _, _ = calc_energy_and_play(
            int(st.session_state[k(cohort, "minutes_per_energy")]),
            bool(st.session_state[k(cohort, "free_5_energy")]),
            int(st.session_state[k(cohort, "buy_10_energy")]),
            int(st.session_state[k(cohort, "buy_20_energy")]),
        )
        st.write(f"하루 최대 플레이: **{dm}회**")
        cur_main = int(st.session_state[k(cohort, "main_play")])
        cur_main = max(0, min(cur_main, dm))
        st.session_state[k(cohort, "main_play")] = cur_main
        st.slider("메인 직접 플레이(회)", 0, max(0, dm), key=k(cohort, "main_play"))
        st.write(f"빠른 소탕(회): **{dm - int(st.session_state[k(cohort, 'main_play')])}**")

        st.divider()
        st.markdown("**소탕 카드(코호트별)**")
        st.checkbox("소탕 카드 ON", key=k(cohort, "sweep_card_on"))
      
        st.divider()
        st.markdown("**골드 슬롯(코호트별)**")
        st.checkbox("에픽 카드(영구) 구매", key=k(cohort, "epic_card_on"))
        st.checkbox("골드 슬롯 카드(월간) 구매", key=k(cohort, "gold_slot_monthly_on"))
        monthly_on = bool(st.session_state[k(cohort, "gold_slot_monthly_on")])
        if monthly_on:
            st.radio("배팅 선택", ["배팅 없음", "2배 배팅", "4배 배팅"], index=0, key=k(cohort, "bet_choice"))
            st.number_input("유료 칩 스핀(일) (0~20, 1칩=20다이아=1스핀)", min_value=0, max_value=20, step=1, key=k(cohort, "chips_used"))
        else:
            st.session_state[k(cohort, "bet_choice")] = "월간 미구매(고정)"
            st.session_state[k(cohort, "chips_used")] = 0
            st.write("월간 미구매: 배팅 2배/4배 불가, 고정 배팅(미구매 1배)만 적용")

        st.divider()
        st.markdown("**자동 보정(코호트별)**")
        low = float(st.session_state["target_low"])
        high = float(st.session_state["target_high"])
        st.caption(f"추천 목표: ratio(잔액/누적골드) = {low:.0%} ~ {high:.0%}")
        if st.button(f"{cohort} 추천 XP 배율 계산", key=k(cohort, "btn_reco")):
            scenario, chosen = recommend_auto_multiplier_for_cohort(cohort)
            st.session_state[k(cohort, "auto_reco_info")] = None
            if chosen is not None:
                st.session_state[k(cohort, "auto_xp_mult_reco")] = float(chosen["m"])
                st.session_state[k(cohort, "auto_reco_info")] = {
                    "scenario": scenario,
                    "m": float(chosen["m"]),
                    "r": float(chosen["r"]),
                    "bal": int(chosen["bal"]),
                    "income": int(chosen["income"]),
                    "lvl": int(chosen["lvl"]),
                }
            st.rerun()

        reco_val = float(st.session_state.get(k(cohort, "auto_xp_mult_reco"), 1.0))
        st.write(f"추천 XP 배율: **{reco_val:.3f}x**")
        if st.button(f"{cohort} 추천 배율을 적용값으로", key=k(cohort, "btn_apply_reco")):
            st.session_state[k(cohort, "auto_xp_mult_apply")] = reco_val
            st.rerun()
        st.slider("추천값 보정 XP 배율(적용값)", 0.10, 20.0, step=0.01, key=k(cohort, "auto_xp_mult_apply"))

# =========================================================
# Render
# =========================================================
def render_result(cohort: str, title: str, sim: Dict[str, Any], auto_mult: float, manual_on: bool):
    df_full = sim["df_full"]
    df_view = sim["df_view"]
    final = sim["final"]
    A = sim["A"]
    B = sim["B"]
    xp_bd = sim["xp_breakdown"]
    gold_bd = sim["gold_breakdown"]
    checks = sim["checks"]
    ops = sim["ops"]

    scenario_target = "A" if cohort == "핵과금" else "B"
    target_balance = A["balance"] if scenario_target == "A" else B["balance"]
    target_ratio = balance_ratio(target_balance, final["gold"])
    target_label = target_classify(target_balance, final["gold"])

    st.subheader(title)

    kpis = st.columns(6)
    kpis[0].metric("현재 시점", f"{int(view_day)}일차 / {int(simulation_days)}일")
    kpis[1].metric("도달 스테이지", f"{final['stage']} / {MAX_STAGE}")
    kpis[2].metric("도달 계정 레벨", f"Lv.{final['level']}")
    kpis[3].metric("누적 XP", f"{final['xp']:,}")
    kpis[4].metric("누적 Gold(수급)", f"{final['gold']:,}")
    kpis[5].metric("목표 판정", status_badge(target_label))

    low = float(st.session_state["target_low"])
    high = float(st.session_state["target_high"])
    st.caption(
        f"코호트: {cohort} | 목표 시나리오: {scenario_target} | "
        f"추천 목표 ratio: {low:.0%}~{high:.0%} | "
        f"현재 ratio: {target_ratio:.2%} | "
    )
    # 실제 적용: manual_on 탭이면서, 해당 코호트가 sweep_card_on일 때만
    sweep_on = bool(st.session_state[k(cohort, "sweep_card_on")])
    manual_mult = (float(st.session_state["xp_boost_N"]) if (manual_on and sweep_on) else 1.0)
    final_xp_mult = float(auto_mult) * manual_mult

    st.caption(
        f"XP 배율 구성 | "
        f"추천값 보정: {auto_mult:.3f}x | "
        f"추가 XP(소탕카드 유저만): {manual_mult:.2f}x | "
        f"최종 적용: {final_xp_mult:.2f}x"
    )


    # 더 알아듣기 쉬운 메시지로 변경(요구사항 반영)
    xp_msg = "정상" if checks["xp_ok"] else "불일치(계산 확인 필요)"
    gold_msg = "정상" if checks["gold_ok"] else "불일치(계산 확인 필요)"
    play_msg = "정상" if checks["play_ok"] else "불일치(메인+소탕 합 확인 필요)"
    st.info(f"계산 점검(현재 시점): XP 합계={xp_msg} · Gold 합계={gold_msg} · 플레이 분배={play_msg}")

    st.caption(
        f"스테이지 정책: 시작 {ops['start_stage']} → {MAX_STAGE} (도달 기간 {ops['days_to_max_stage']}일, alpha {ops['alpha']:.2f}) | "
        f"하루 플레이 {ops['daily_max_play']}회(메인 {ops['main_play']}/빠른소탕 {ops['quick_sweep']}) | "
        f"골드 슬롯 스핀 {ops['total_spins']}회(무료 {ops['free_spins']}/유료 {ops['paid_spins']}) | "
        f"일일 다이아(가정) {ops['daily_diamond']:,}"
    )

    g1, g2, g3 = st.columns([1, 1, 1])
    with g1:
        st.markdown("**스테이지(일별)**")
        st.line_chart(df_full.set_index("day")["stage"])
    with g2:
        st.markdown("**계정 레벨(일별)**")
        st.line_chart(df_full.set_index("day")["account_level"])
    with g3:
        st.markdown("**누적 골드(일별)**")
        st.line_chart(df_full.set_index("day")["cum_gold"])

    st.divider()

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("### 경험치 획득처(누적/비율) — 현재 시점까지")
        st.dataframe(xp_bd, use_container_width=True, hide_index=True)
    with b2:
        st.markdown("### 골드 획득처(누적/비율) — 현재 시점까지")
        st.dataframe(gold_bd, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("## 소비 가정(A/B) 결과 — 현재 시점까지")
    if use_gear_table and gear_level_max > 0:
        # 문구를 더 직관적으로 변경(요구사항 반영)
        st.info(
            f"장비 강화 골드 소비(누적, 가정): **{final['gear_spend']:,}**  |  "
            f"추정 장비 레벨(계정레벨 기반): **{final['gear_level']}**"
        )

    cA, cB = st.columns(2)
    with cA:
        st.markdown("### 가정 A")
        st.caption("최종 레벨이 속한 5레벨 구간의 베이스 레벨까지 해금된 패시브를 전부 뽑음")
        st.write(f"- 패시브 기준 레벨: **Lv.{A['base']}**")
        st.write(f"- 패시브 소비: **{A['passive_spend']:,}** (draws {A['draws']})")
        st.write(f"- 총 소비(패시브+장비): **{A['total_spend']:,}**")
        st.write(f"- 잔액: **{A['balance']:,}** (ratio {balance_ratio(A['balance'], final['gold']):.2%})")
        st.write(f"- 목표 판정: **{status_badge(target_classify(A['balance'], final['gold']))}**")

    with cB:
        st.markdown("### 가정 B (변경된 모델)")
        st.caption("5레벨 구간의 진행도에 따라 '다음 해금 구간' 패시브를 일부→거의→완료로 점진 반영")
        st.write(f"- 완료 구간(전부 구매): **Lv.{B['completed_cap']}까지**")
        if B.get("partial_fraction", 0.0) > 0:
            st.write(f"- 진행 구간(부분 구매): **Lv.{B['partial_cap']}까지** 중 약 **{B['partial_fraction']:.0%}** 반영")
        else:
            st.write(f"- 진행 구간(부분 구매): **없음(완료 구간까지만 반영)**")
        st.write(f"- 패시브 소비: **{B['passive_spend']:,}** (draws {B['draws']})")
        st.write(f"- 총 소비(패시브+장비): **{B['total_spend']:,}**")
        st.write(f"- 잔액: **{B['balance']:,}** (ratio {balance_ratio(B['balance'], final['gold']):.2%})")
        st.write(f"- 목표 판정: **{status_badge(target_classify(B['balance'], final['gold']))}**")

# =========================================================
# Run + Render
# =========================================================
cohort_tabs = st.tabs([f"{c}" for c in COHORTS])

for i, cohort in enumerate(COHORTS):
    with cohort_tabs[i]:
        manual_sim = run_simulation_for_cohort(cohort, auto_xp_mult=1.0, use_manual=True)
        applied_auto = float(st.session_state.get(k(cohort, "auto_xp_mult_apply"), 1.0))
        reco_sim = run_simulation_for_cohort(cohort, auto_xp_mult=applied_auto, use_manual=False)

        if manual_sim is None or reco_sim is None:
            st.error("시뮬레이션 결과가 비어 있습니다. 데이터/입력값을 확인해주세요.")
            continue

        inner_tabs = st.tabs(["1) 수동 값 적용", "2) 추천값 적용"])
        with inner_tabs[0]:
            render_result(cohort, "1) 수동 값 적용", manual_sim, auto_mult=1.0, manual_on=True)
        with inner_tabs[1]:
            render_result(cohort, "2) 추천값 적용", reco_sim, auto_mult=applied_auto, manual_on=False)
