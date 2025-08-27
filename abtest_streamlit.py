# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

st.set_page_config(page_title="A/B Test Dashboard + MAB", layout="wide")

# -----------------------------
# 공통 유틸
# -----------------------------
def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan)

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Date 정리(있을 경우)
    for c in ["Date","DATE","date","ds","Day","day","DATE_TZ"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c])
                df = df.sort_values(c).rename(columns={c: "Date"})
                break
            except Exception:
                pass

    # 파생 지표 생성
    if "CTR" not in df.columns and {"# of Website Clicks", "# of Impressions"} <= set(df.columns):
        df["CTR"] = safe_div(df["# of Website Clicks"], df["# of Impressions"])
    if "CVR" not in df.columns and {"# of Purchase", "# of Website Clicks"} <= set(df.columns):
        df["CVR"] = safe_div(df["# of Purchase"], df["# of Website Clicks"])
    if "Revenue" not in df.columns:
        if "Revenue_lognormal" in df.columns:
            df["Revenue"] = pd.to_numeric(df["Revenue_lognormal"], errors="coerce")
        elif {"Spend [USD]", "ROAS"} <= set(df.columns):
            df["Revenue"] = pd.to_numeric(df["Spend [USD]"], errors="coerce") * pd.to_numeric(df.get("ROAS"), errors="coerce")
    if "ROAS" not in df.columns and {"Revenue","Spend [USD]"} <= set(df.columns):
        df["ROAS"] = safe_div(df["Revenue"], df["Spend [USD]"])
    if "Frequency" not in df.columns and {"# of Impressions","Reach"} <= set(df.columns):
        df["Frequency"] = safe_div(df["# of Impressions"], df["Reach"])
    return df

def welch(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().values
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().values
    t, p = stats.ttest_ind(a, b, equal_var=False)

    n1, n2 = len(a), len(b)
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)

    sp = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2)) if (n1+n2-2) > 0 else np.nan
    d  = (m1 - m2) / sp if (sp and sp > 0) else np.nan
    J  = 1 - (3/(4*(n1+n2) - 9)) if (n1+n2) > 2 else 1.0
    g  = d * J

    se     = np.sqrt(v1/n1 + v2/n2) if (n1>1 and n2>1) else np.nan
    df_num = (v1/n1 + v2/n2)**2
    df_den = (v1**2/(n1**2*(n1-1))) + (v2**2/(n2**2*(n2-1))) if (n1>1 and n2>1) else np.nan
    dof    = df_num/df_den if isinstance(df_den, (float, np.floating)) and df_den>0 else np.nan
    if np.isfinite(se) and np.isfinite(dof):
        crit = stats.t.ppf(0.975, dof)  # 95%
        ci_low, ci_high = (m1-m2) - crit*se, (m1-m2) + crit*se
    else:
        ci_low = ci_high = np.nan

    return dict(
        mean_test=m1, mean_control=m2,
        diff=m1-m2, p_value=p, t_stat=t,
        hedges_g=g, ci_low=ci_low, ci_high=ci_high
    )

RNG = np.random.default_rng(7)

def get_true_means(control, test, metric):
    mu_c = float(pd.to_numeric(control[metric], errors="coerce").mean(skipna=True))
    mu_t = float(pd.to_numeric(test[metric], errors="coerce").mean(skipna=True))
    return {"Control": mu_c, "Test": mu_t}

# -----------------------------
# Click=100% 퍼널 전환율용 자동 컬럼 탐색 (강화판)
# -----------------------------
import re

CLICK_CANDS    = ["# of Website Clicks", "Clicks", "Click", "Total Clicks",
                  "websiteclicks", "totalclicks"]
CONTENT_CANDS  = ["Content", "View Content", "ViewContent", "Content Views", "View_Content",
                  "viewcontent", "contentviews", "contentview", "view_contents", "content_view", "content"]
CART_CANDS     = ["Cart", "Add to Cart", "AddToCart", "ATC", "Cart Adds", "Add_To_Cart",
                  "addtocart", "cartadds", "addcart", "cartadd", "atc", "cart"]
PURCHASE_CANDS = ["# of Purchase", "Purchases", "Purchase", "Orders",
                  "purchase", "purchases", "order", "orders"]

def _norm(s: str) -> str:
    # 소문자 + 영숫자만 남김
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _pick_first(df: pd.DataFrame, cand_list):
    norm_map = {_norm(col): col for col in df.columns}
    # 1) 정확 일치
    for cand in cand_list:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    # 2) 부분 포함
    norm_cols = list(norm_map.keys())
    for cand in cand_list:
        k = _norm(cand)
        for nc in norm_cols:
            if k and (k in nc or nc in k):
                return norm_map[nc]
    # 3) 원문 일치
    for cand in cand_list:
        if cand in df.columns:
            return cand
    return None

def _resolve_stages(df):
    cols = {
        "clicks":   _pick_first(df, CLICK_CANDS),
        "content":  _pick_first(df, CONTENT_CANDS),
        "cart":     _pick_first(df, CART_CANDS),
        "purchase": _pick_first(df, PURCHASE_CANDS),
    }
    missing = [k for k, v in cols.items() if v is None]
    return cols, missing

def _click_based_rates(df):
    cols, missing = _resolve_stages(df)
    if missing:
        return None, missing
    for c in cols.values():
        df[c] = pd.to_numeric(df[c], errors="coerce")
    s_click = pd.to_numeric(df[cols["clicks"]], errors="coerce").sum(skipna=True)
    if not np.isfinite(s_click) or s_click <= 0:
        return None, ["clicks_sum<=0"]
    rates = {
        "Click":    100.0,
        "Content":  float(pd.to_numeric(df[cols["content"]],  errors="coerce").sum(skipna=True)  / s_click * 100.0),
        "Cart":     float(pd.to_numeric(df[cols["cart"]],     errors="coerce").sum(skipna=True)  / s_click * 100.0),
        "Purchase": float(pd.to_numeric(df[cols["purchase"]], errors="coerce").sum(skipna=True)  / s_click * 100.0),
    }
    return rates, []

# -----------------------------
# MAB 시뮬레이터
# -----------------------------
def simulate_epsilon_greedy(true_means, n_rounds, epsilon, bernoulli=True, variance=0.1):
    arms = list(true_means.keys())
    counts = {a: 0 for a in arms}
    sums   = {a: 0.0 for a in arms}
    total, cum = 0.0, []
    for _ in range(n_rounds):
        if RNG.random() < epsilon:
            a = RNG.choice(arms)
        else:
            avg = {arm: (sums[arm]/counts[arm]) if counts[arm]>0 else 0.0 for arm in arms}
            a = max(avg, key=avg.get)
        mu = true_means[a]
        if bernoulli:
            r = float(RNG.random() < mu)
        else:
            r = float(max(RNG.normal(mu, np.sqrt(variance)), 0.0))
        counts[a]+=1; sums[a]+=r; total+=r; cum.append(total)
    return np.array(cum)

def simulate_thompson_bernoulli(true_means, n_rounds):
    arms = list(true_means.keys())
    alpha = {a: 1.0 for a in arms}
    beta  = {a: 1.0 for a in arms}
    total, cum = 0.0, []
    for _ in range(n_rounds):
        theta = {a: RNG.beta(alpha[a], beta[a]) for a in arms}
        a = max(theta, key=theta.get)
        r = float(RNG.random() < true_means[a])
        alpha[a] += r; beta[a] += (1.0-r)
        total += r; cum.append(total)
    return np.array(cum)

def simulate_thompson_gaussian(true_means, n_rounds, obs_var=0.15):
    arms = list(true_means.keys())
    n = {a: 0 for a in arms}
    m = {a: 0.0 for a in arms}
    total, cum = 0.0, []
    for _ in range(n_rounds):
        theta = {a: RNG.normal(m[a], np.sqrt(obs_var/(n[a]+1))) for a in arms}
        a = max(theta, key=theta.get)
        r = float(max(RNG.normal(true_means[a], np.sqrt(obs_var)), 0.0))
        n[a]+=1; m[a]+= (r - m[a]) / n[a]
        total += r; cum.append(total)
    return np.array(cum)

def simulate_ab_fixed(true_means, n_rounds, bernoulli=True, variance=0.1):
    arms = list(true_means.keys())
    total, cum = 0.0, []
    for t in range(n_rounds):
        a = arms[t % 2]
        mu = true_means[a]
        if bernoulli:
            r = float(RNG.random() < mu)
        else:
            r = float(max(RNG.normal(mu, np.sqrt(variance)), 0.0))
        total += r; cum.append(total)
    return np.array(cum)

# -----------------------------
# 데이터 로딩
# -----------------------------
CTRL_PATH = Path("control.csv")
TEST_PATH = Path("test.csv")
if not CTRL_PATH.exists() or not TEST_PATH.exists():
    st.error("control.csv / test.csv 파일이 필요합니다.")
    st.stop()

control = prepare_df(pd.read_csv(CTRL_PATH))
test    = prepare_df(pd.read_csv(TEST_PATH))

# -----------------------------
# 대시보드 시작
# -----------------------------
st.title("A/B Test Dashboard + MAB")

cols = st.columns(5)
for i, label in enumerate(["Revenue","ROAS","Frequency","CTR","CVR"]):
    if label in control.columns and label in test.columns:
        c = float(pd.to_numeric(control[label], errors="coerce").mean())
        t = float(pd.to_numeric(test[label], errors="coerce").mean())
        delta = (t - c) / c * 100 if c else np.nan
        cols[i].metric(label, f"{t:,.4g}", f"{delta:+.1f}% vs Control")

# -----------------------------
# 1) Welch t-test
# -----------------------------
st.header("1) 기본 가설 검정 (Welch t-test)")
metrics_order = ["Revenue","ROAS","Frequency","CTR","CVR"]
rows = []
for m in metrics_order:
    if m in control.columns and m in test.columns:
        r = welch(test[m], control[m])
        rows.append(dict(Metric=m,
                         **{"Control mean":r["mean_control"], "Test mean":r["mean_test"]},
                         **{"Δ(Test-Control)":r["diff"], "p-value":r["p_value"], "Hedges g":r["hedges_g"],
                            "CI low":r["ci_low"], "CI high":r["ci_high"]}))
res_df = pd.DataFrame(rows)
if not res_df.empty:
    res_df["Sig"] = np.where(res_df["p-value"] < 0.05, "유의미", "")

    def _highlight_sig(row):
        return [("background-color:#1f5130" if row["Sig"]=="유의미" else "")]*len(row)

    st.dataframe(
        res_df[["Metric","Control mean","Test mean","Δ(Test-Control)","p-value","Hedges g","CI low","CI high","Sig"]]
        .style.format({
            "Control mean":"{:.6g}","Test mean":"{:.6g}",
            "Δ(Test-Control)":"{:.6g}","p-value":"{:.4g}",
            "Hedges g":"{:.3g}","CI low":"{:.6g}","CI high":"{:.6g}"
        }).apply(_highlight_sig, axis=1),
        use_container_width=True
    )
else:
    st.info("겹치는 지표가 없어 t-test 표를 표시하지 않습니다.")

# -----------------------------
# 2) 퍼널 분석
# -----------------------------
st.header("2) 퍼널 분석")

st.subheader("2-A) Funnel Conversion (Click = 100%)")
ctrl_rates, miss1 = _click_based_rates(control)
test_rates, miss2 = _click_based_rates(test)

if miss1 or miss2:
    st.warning(
        "클릭 기준 퍼널을 위한 컬럼을 찾지 못했습니다.\n"
        f"- Control 누락: {miss1 or '없음'}\n- Test 누락: {miss2 or '없음'}\n"
        f"지원 후보:\n"
        f"  Clicks: {CLICK_CANDS}\n  Content: {CONTENT_CANDS}\n"
        f"  Cart: {CART_CANDS}\n  Purchase: {PURCHASE_CANDS}"
    )
else:
    stages_c = ["Click","Content","Cart","Purchase"]
    ctrl_y   = [ctrl_rates[s] for s in stages_c]
    test_y   = [test_rates[s] for s in stages_c]

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Bar(name="Control", x=stages_c, y=ctrl_y,
                            text=[f"{v:.2f}%" for v in ctrl_y], textposition="outside"))
    fig_fc.add_trace(go.Bar(name="Test",    x=stages_c, y=test_y,
                            text=[f"{v:.2f}%" for v in test_y], textposition="outside"))
    ymax = max(max(ctrl_y), max(test_y))
    fig_fc.update_layout(
        barmode="group",
        height=430,
        title="Funnel Conversion (Click = 100%)",
        yaxis=dict(title="Conversion Rate (%)", range=[0, ymax*1.25])
    )
    st.plotly_chart(fig_fc, use_container_width=True)

# -----------------------------
# 3) 멀티암 밴딧 시뮬레이터 (Revenue & ROAS 중심)
# -----------------------------
st.header("3) 멀티암 밴딧 시뮬레이터")
with st.expander("시뮬레이션 옵션", expanded=True):
    n_rounds = st.slider("Rounds", 200, 5000, 2000, 100)
    epsilon  = st.slider("ε (epsilon-greedy)", 0.00, 0.50, 0.10, 0.01)

# 탭 구성: Revenue, ROAS, CTR(참고)
tab_rev, tab_roas, tab_ctr = st.tabs(["Revenue", "ROAS", "CTR (Clicks)"])

# Revenue (연속형 보상)
with tab_rev:
    if {"Revenue"} <= set(control.columns) and {"Revenue"} <= set(test.columns):
        true_rev = get_true_means(control, test, "Revenue")
        # 연속형 보상 시뮬레이션: ε-greedy / Thompson Gaussian / 고정 A/B
        rev_eps = simulate_epsilon_greedy(true_rev, n_rounds, epsilon, bernoulli=False, variance=0.20)
        rev_ts  = simulate_thompson_gaussian(true_rev, n_rounds, obs_var=0.20)
        rev_ab  = simulate_ab_fixed(true_rev, n_rounds, bernoulli=False, variance=0.20)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=rev_eps, x=np.arange(n_rounds), name="ε-greedy"))
        fig.add_trace(go.Scatter(y=rev_ts,  x=np.arange(n_rounds), name="Thompson"))
        fig.add_trace(go.Scatter(y=rev_ab,  x=np.arange(n_rounds), name="A/B (50:50)", line=dict(dash="dash")))
        fig.update_layout(height=380, title="Cumulative Revenue (simulated)", xaxis_title="Round", yaxis_title="Cumulative")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Control Revenue ≈ {true_rev['Control']:.3g}, Test Revenue ≈ {true_rev['Test']:.3g}")
    else:
        st.info("Revenue 컬럼이 양 실험군 모두에 있어야 Revenue 시뮬레이션을 그릴 수 있습니다.")

# ROAS (연속형 보상)
with tab_roas:
    if {"ROAS"} <= set(control.columns) and {"ROAS"} <= set(test.columns):
        true_roas = get_true_means(control, test, "ROAS")
        roas_eps = simulate_epsilon_greedy(true_roas, n_rounds, epsilon, bernoulli=False, variance=0.15)
        roas_ts  = simulate_thompson_gaussian(true_roas, n_rounds, obs_var=0.15)
        roas_ab  = simulate_ab_fixed(true_roas, n_rounds, bernoulli=False, variance=0.15)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=roas_eps, x=np.arange(n_rounds), name="ε-greedy"))
        fig.add_trace(go.Scatter(y=roas_ts,  x=np.arange(n_rounds), name="Thompson"))
        fig.add_trace(go.Scatter(y=roas_ab,  x=np.arange(n_rounds), name="A/B (50:50)", line=dict(dash="dash")))
        fig.update_layout(height=380, title="Cumulative ROAS (simulated)", xaxis_title="Round", yaxis_title="Cumulative")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Control ROAS ≈ {true_roas['Control']:.3g}, Test ROAS ≈ {true_roas['Test']:.3g}")
    else:
        st.info("ROAS 컬럼이 양 실험군 모두에 있어야 ROAS 시뮬레이션을 그릴 수 있습니다.")

# CTR (참고용, 베르누이 보상)
with tab_ctr:
    if {"CTR"} <= set(control.columns) and {"CTR"} <= set(test.columns):
        true_ctr = get_true_means(control, test, "CTR")
        ctr_eps = simulate_epsilon_greedy(true_ctr, n_rounds, epsilon, bernoulli=True)
        ctr_ts  = simulate_thompson_bernoulli(true_ctr, n_rounds)
        ctr_ab  = simulate_ab_fixed(true_ctr, n_rounds, bernoulli=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=ctr_eps, x=np.arange(n_rounds), name="ε-greedy"))
        fig.add_trace(go.Scatter(y=ctr_ts,  x=np.arange(n_rounds), name="Thompson"))
        fig.add_trace(go.Scatter(y=ctr_ab,  x=np.arange(n_rounds), name="A/B (50:50)", line=dict(dash="dash")))
        fig.update_layout(height=380, title="Cumulative Clicks", xaxis_title="Round", yaxis_title="Cumulative")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Control CTR ≈ {true_ctr['Control']:.4f}, Test CTR ≈ {true_ctr['Test']:.4f}")
    else:
        st.info("CTR 컬럼이 양 실험군 모두에 있을 때만 CTR 시뮬레이션을 표시합니다.")

# -----------------------------
# 4) 시계열 비교
# -----------------------------
st.header("4) 시계열 비교")
if "Date" in control.columns and "Date" in test.columns:
    metric_ts = st.selectbox(
        "시계열로 볼 지표",
        [m for m in ["CTR","CVR","Revenue","ROAS","# of Impressions","# of Website Clicks","# of Purchase"]
         if m in control.columns and m in test.columns]
    )
    ctrl_daily = control.groupby("Date")[metric_ts].mean().rename("Control")
    test_daily = test.groupby("Date")[metric_ts].mean().rename("Test")
    ts = pd.concat([ctrl_daily, test_daily], axis=1).reset_index()
    fig = px.line(ts, x="Date", y=["Control","Test"], title=f"Daily {metric_ts}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("시계열 그래프는 Date 컬럼이 있을 때 표시됩니다.")
