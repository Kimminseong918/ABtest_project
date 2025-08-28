# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import re

# 페이지 설정을 맨 위로 이동
st.set_page_config(page_title="A/B Test Dashboard + MAB", layout="wide")

# -----------------------------
# 공통 유틸 및 데이터 처리
# -----------------------------

# Streamlit 캐시를 사용하여 데이터 로딩 및 전처리 속도 향상
@st.cache_data
def load_and_prepare_df(source) -> pd.DataFrame:
    """CSV 파일을 로드하고 공통 전처리 단계를 적용합니다."""
    df = pd.read_csv(source)
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Date 컬럼 자동 감지 및 변환
    for c in ["Date", "DATE", "date", "ds", "Day", "day", "DATE_TZ"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c])
                df = df.sort_values(c).rename(columns={c: "Date"})
                break
            except Exception:
                pass

    # 파생 지표 생성
    def safe_div(a, b):
        return (a / b).replace([np.inf, -np.inf], np.nan)

    if "CTR" not in df.columns and {"# of Website Clicks", "# of Impressions"} <= set(df.columns):
        df["CTR"] = safe_div(pd.to_numeric(df["# of Website Clicks"], errors="coerce"), pd.to_numeric(df["# of Impressions"], errors="coerce"))
    if "CVR" not in df.columns and {"# of Purchase", "# of Website Clicks"} <= set(df.columns):
        df["CVR"] = safe_div(pd.to_numeric(df["# of Purchase"], errors="coerce"), pd.to_numeric(df["# of Website Clicks"], errors="coerce"))
    if "Revenue" not in df.columns:
        if "Revenue_lognormal" in df.columns:
            df["Revenue"] = pd.to_numeric(df["Revenue_lognormal"], errors="coerce")
        elif {"Spend [USD]", "ROAS"} <= set(df.columns):
            df["Revenue"] = pd.to_numeric(df["Spend [USD]"], errors="coerce") * pd.to_numeric(df.get("ROAS"), errors="coerce")
    if "ROAS" not in df.columns and {"Revenue", "Spend [USD]"} <= set(df.columns):
        df["ROAS"] = safe_div(pd.to_numeric(df["Revenue"], errors="coerce"), pd.to_numeric(df["Spend [USD]"], errors="coerce"))
    if "Frequency" not in df.columns and {"# of Impressions", "Reach"} <= set(df.columns):
        df["Frequency"] = safe_div(pd.to_numeric(df["# of Impressions"], errors="coerce"), pd.to_numeric(df["Reach"], errors="coerce"))

    return df

def welch(a, b):
    """두 샘플에 대한 Welch's t-test를 수행하고 관련 통계량을 반환합니다."""
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().values
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().values

    if len(a) < 2 or len(b) < 2: # 샘플 수가 부족하면 계산 불가
        return dict(mean_test=np.nan, mean_control=np.nan, diff=np.nan, p_value=np.nan, t_stat=np.nan,
                    hedges_g=np.nan, ci_low=np.nan, ci_high=np.nan)

    t, p = stats.ttest_ind(a, b, equal_var=False)

    n1, n2 = len(a), len(b)
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)

    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    d = (m1 - m2) / sp if sp > 0 else 0.0
    J = 1 - (3 / (4 * (n1 + n2) - 9))
    g = d * J

    se = np.sqrt(v1 / n1 + v2 / n2)
    df_num = (v1 / n1 + v2 / n2)**2
    df_den = (v1**2 / (n1**2 * (n1 - 1))) + (v2**2 / (n2**2 * (n2 - 1)))
    dof = df_num / df_den if df_den > 0 else 0

    if np.isfinite(se) and dof > 0:
        crit = stats.t.ppf(0.975, dof)  # 95%
        ci_low, ci_high = (m1 - m2) - crit * se, (m1 - m2) + crit * se
    else:
        ci_low = ci_high = np.nan

    return dict(mean_test=m1, mean_control=m2, diff=m1 - m2, p_value=p, t_stat=t,
                hedges_g=g, ci_low=ci_low, ci_high=ci_high)

RNG = np.random.default_rng(7) # 재현성을 위한 시드 고정

def get_true_means(control, test, metric):
    mu_c = float(pd.to_numeric(control[metric], errors="coerce").mean(skipna=True))
    mu_t = float(pd.to_numeric(test[metric], errors="coerce").mean(skipna=True))
    return {"Control": mu_c, "Test": mu_t}

# -----------------------------
# 퍼널 분석용 함수
# -----------------------------
CLICK_CANDS = ["# of Website Clicks", "Clicks", "Click", "Total Clicks", "websiteclicks", "totalclicks"]
CONTENT_CANDS = ["Content", "View Content", "ViewContent", "Content Views", "View_Content", "viewcontent", "contentviews"]
CART_CANDS = ["Cart", "Add to Cart", "AddToCart", "ATC", "Cart Adds", "Add_To_Cart", "addtocart"]
PURCHASE_CANDS = ["# of Purchase", "Purchases", "Purchase", "Orders", "purchase", "orders"]

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _pick_first(df: pd.DataFrame, cand_list):
    norm_map = {_norm(col): col for col in df.columns}
    norm_cols = list(norm_map.keys())
    for cand in cand_list:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
        for nc in norm_cols:
            if key and (key in nc or nc in key):
                return norm_map[nc]
    return None

def _click_based_rates(df):
    cols = {
        "clicks": _pick_first(df, CLICK_CANDS),
        "content": _pick_first(df, CONTENT_CANDS),
        "cart": _pick_first(df, CART_CANDS),
        "purchase": _pick_first(df, PURCHASE_CANDS),
    }
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        return None, missing

    for c in cols.values():
        df[c] = pd.to_numeric(df[c], errors="coerce")

    s_click = df[cols["clicks"]].sum(skipna=True)
    if not np.isfinite(s_click) or s_click <= 0:
        return None, ["clicks_sum<=0"]

    rates = {
        "Click": 100.0,
        "Content": float(df[cols["content"]].sum(skipna=True) / s_click * 100.0),
        "Cart": float(df[cols["cart"]].sum(skipna=True) / s_click * 100.0),
        "Purchase": float(df[cols["purchase"]].sum(skipna=True) / s_click * 100.0),
    }
    return rates, []

# -----------------------------
# 시뮬레이터 함수
# -----------------------------
def simulate_epsilon_greedy(true_means, n_rounds, epsilon, variance=0.1):
    arms, cum = list(true_means.keys()), [0.0] * n_rounds
    counts = {a: 0 for a in arms}; sums = {a: 0.0 for a in arms}
    for i in range(n_rounds):
        if RNG.random() < epsilon:
            a = RNG.choice(arms)
        else:
            avg = {arm: (sums[arm] / counts[arm]) if counts[arm] > 0 else 0.0 for arm in arms}
            a = max(avg, key=avg.get)
        r = float(max(RNG.normal(true_means[a], np.sqrt(variance)), 0.0))
        counts[a] += 1; sums[a] += r
        cum[i] = cum[i-1] + r if i > 0 else r
    return np.array(cum)

def simulate_thompson_gaussian(true_means, n_rounds, obs_var=0.15):
    arms, cum = list(true_means.keys()), [0.0] * n_rounds
    n = {a: 0 for a in arms}; m = {a: 0.0 for a in arms}
    for i in range(n_rounds):
        theta = {a: RNG.normal(m[a], np.sqrt(obs_var / (n[a] + 1))) for a in arms}
        a = max(theta, key=theta.get)
        r = float(max(RNG.normal(true_means[a], np.sqrt(obs_var)), 0.0))
        n[a] += 1; m[a] += (r - m[a]) / n[a]
        cum[i] = cum[i-1] + r if i > 0 else r
    return np.array(cum)

def simulate_ab_fixed(true_means, n_rounds, variance=0.1):
    arms, cum = list(true_means.keys()), [0.0] * n_rounds
    for t in range(n_rounds):
        a = arms[t % 2]
        r = float(max(RNG.normal(true_means[a], np.sqrt(variance)), 0.0))
        cum[t] = cum[t-1] + r if t > 0 else r
    return np.array(cum)

# =============================
# 대시보드 UI 시작
# =============================
st.title("A/B Test Dashboard + MAB Simulator")

# --- 사이드바: 파일 업로드 ---
st.sidebar.header("데이터 업로드")
uploaded_control = st.sidebar.file_uploader("Upload Control Group CSV", type="csv")
uploaded_test = st.sidebar.file_uploader("Upload Test Group CSV", type="csv")

control, test = None, None
if uploaded_control and uploaded_test:
    control = load_and_prepare_df(uploaded_control)
    test = load_and_prepare_df(uploaded_test)
else:
    # 로컬 파일이 있으면 폴백으로 사용
    CTRL_PATH, TEST_PATH = Path("control.csv"), Path("test.csv")
    if CTRL_PATH.exists() and TEST_PATH.exists():
        control = load_and_prepare_df(CTRL_PATH)
        test = load_and_prepare_df(TEST_PATH)
    else:
        st.warning("👈 사이드바에서 Control과 Test 그룹의 CSV 파일을 업로드해주세요.")
        st.stop()

# --- 대시보드 메인 ---
st.header("📈 요약 지표")
cols = st.columns(5)
metrics_summary = ["Revenue", "ROAS", "Frequency", "CTR", "CVR"]
for i, label in enumerate(metrics_summary):
    if label in control.columns and label in test.columns:
        c = pd.to_numeric(control[label], errors="coerce").mean()
        t = pd.to_numeric(test[label], errors="coerce").mean()
        delta = (t - c) / c * 100 if c and c != 0 else np.nan
        cols[i].metric(label, f"{t:,.4g}", f"{delta:+.1f}% vs Control")

# -----------------------------
# 1) Welch t-test
# -----------------------------
st.header("1) 가설 검정 (Welch's t-test)")
rows = []
for m in ["Revenue", "ROAS", "Frequency", "CTR", "CVR"]:
    if m in control.columns and m in test.columns:
        r = welch(test[m], control[m])
        row_data = {"Metric": m, "Control mean": r["mean_control"], "Test mean": r["mean_test"],
                    "Δ(Test-Control)": r["diff"], "p-value": r["p_value"], "Hedges g": r["hedges_g"],
                    "CI low": r["ci_low"], "CI high": r["ci_high"]}
        rows.append(row_data)

if rows:
    res_df = pd.DataFrame(rows)
    res_df["Sig"] = np.where(res_df["p-value"] < 0.05, "유의미", "")
    st.dataframe(
        res_df.style.format({
            "Control mean": "{:.4g}", "Test mean": "{:.4g}", "Δ(Test-Control)": "{:.4g}",
            "p-value": "{:.3f}", "Hedges g": "{:.3f}", "CI low": "{:.4g}", "CI high": "{:.4g}"
        }).apply(lambda row: [("background-color:#1f5130" if row["Sig"] == "유의미" else "")] * len(row), axis=1),
        use_container_width=True
    )
else:
    st.info("공통 지표가 없어 t-test를 수행할 수 없습니다.")

# -----------------------------
# 2) 퍼널 분석
# -----------------------------
st.header("2) 퍼널 분석")
ctrl_rates, miss1 = _click_based_rates(control)
test_rates, miss2 = _click_based_rates(test)

if miss1 or miss2:
    st.warning(
        "클릭 기준 퍼널을 위한 컬럼을 찾지 못했습니다.\n"
        f"- Control 누락: {miss1 or '없음'}\n- Test 누락: {miss2 or '없음'}\n"
        "아래 후보군 중 하나와 일치하는 컬럼 이름이 필요합니다:\n"
        f"  - Clicks: {CLICK_CANDS}\n  - Content: {CONTENT_CANDS}\n"
        f"  - Cart: {CART_CANDS}\n  - Purchase: {PURCHASE_CANDS}"
    )
else:
    stages = ["Click", "Content", "Cart", "Purchase"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Control", x=stages, y=[ctrl_rates[s] for s in stages], text=[f"{v:.2f}%" for v in [ctrl_rates[s] for s in stages]], textposition="outside"))
    fig.add_trace(go.Bar(name="Test", x=stages, y=[test_rates[s] for s in stages], text=[f"{v:.2f}%" for v in [test_rates[s] for s in stages]], textposition="outside"))
    ymax = max(max(v for v in ctrl_rates.values()), max(v for v in test_rates.values()))
    fig.update_layout(barmode="group", height=430, title="Funnel Conversion (Click = 100%)", yaxis_title="Conversion Rate (%)", yaxis_range=[0, ymax * 1.25])
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 3) 멀티암 밴딧 시뮬레이터
# -----------------------------
st.header("3) 멀티암 밴딧 시뮬레이터")

# (개선) MAB 시뮬레이션 및 플롯 생성을 위한 헬퍼 함수
def run_and_plot_mab(metric, control_df, test_df, n_rounds, epsilon):
    if metric not in control_df.columns or metric not in test_df.columns:
        st.info(f"'{metric}' 컬럼이 양쪽 데이터에 모두 있어야 시뮬레이션을 실행할 수 있습니다.")
        return

    true_means = get_true_means(control_df, test_df, metric)
    variance = 0.20 if metric == "Revenue" else 0.15 # 지표별 분산 가정

    # 시뮬레이션 실행
    sim_eps = simulate_epsilon_greedy(true_means, n_rounds, epsilon, variance=variance)
    sim_ts = simulate_thompson_gaussian(true_means, n_rounds, obs_var=variance)
    sim_ab = simulate_ab_fixed(true_means, n_rounds, variance=variance)

    # 그래프 생성
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=sim_eps, name="ε-greedy", mode='lines'))
    fig.add_trace(go.Scatter(y=sim_ts, name="Thompson Sampling", mode='lines'))
    fig.add_trace(go.Scatter(y=sim_ab, name="A/B (50:50)", mode='lines', line=dict(dash="dash")))
    fig.update_layout(height=380, title=f"누적 {metric} 시뮬레이션 결과",
                      xaxis_title="Round", yaxis_title="Cumulative Value", legend_title="Algorithm")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"실제 평균값(추정): Control {metric} ≈ {true_means['Control']:.3g}, Test {metric} ≈ {true_means['Test']:.3g}")

with st.expander("시뮬레이션 옵션", expanded=True):
    n_rounds = st.slider("Rounds", 200, 5000, 2000, 100)
    epsilon = st.slider("ε (epsilon-greedy)", 0.00, 0.50, 0.10, 0.01)

tab_rev, tab_roas = st.tabs(["Revenue", "ROAS"])
with tab_rev:
    run_and_plot_mab("Revenue", control, test, n_rounds, epsilon)
with tab_roas:
    run_and_plot_mab("ROAS", control, test, n_rounds, epsilon)

# -----------------------------
# 4) 시계열 비교
# -----------------------------
st.header("4) 시계열 비교")
if "Date" in control.columns and "Date" in test.columns:
    metric_ts = st.selectbox(
        "시계열로 볼 지표",
        [m for m in ["CTR", "CVR", "Revenue", "ROAS", "# of Impressions", "# of Website Clicks", "# of Purchase"]
         if m in control.columns and m in test.columns]
    )
    if metric_ts:
        ctrl_daily = control.groupby("Date")[metric_ts].mean().rename("Control")
        test_daily = test.groupby("Date")[metric_ts].mean().rename("Test")
        ts = pd.concat([ctrl_daily, test_daily], axis=1).reset_index()
        fig = px.line(ts, x="Date", y=["Control", "Test"], title=f"일별 {metric_ts} 추이")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("시계열 그래프는 양쪽 데이터에 'Date' 컬럼이 모두 있을 때 표시됩니다.")
