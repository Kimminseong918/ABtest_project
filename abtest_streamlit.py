# streamlit_app.py
# -*- coding: utf-8 -*-
import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# --- Matplotlib (있으면 사용, 없으면 Plotly로 폴백) ---
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

st.set_page_config(page_title="A/B Test Dashboard + MAB", layout="wide")

# ----------------------------
# Utilities
# ----------------------------
def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan)

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Date 통일
    for c in ["Date","DATE","date","ds","Day","day","DATE_TZ"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c])
                df = df.sort_values(c).rename(columns={c: "Date"})
                break
            except Exception:
                pass

    # 파생 지표
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

# --- Welch t-test (250827.py 로직) ---
def welch(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().values
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().values

    # Welch t-test
    t, p = stats.ttest_ind(a, b, equal_var=False)

    # 표본 크기/평균/분산
    n1, n2 = len(a), len(b)
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)

    # Hedges' g (pooled SD)
    sp = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2)) if (n1+n2-2) > 0 else np.nan
    d  = (m1 - m2) / sp if (sp and sp > 0) else np.nan
    J  = 1 - (3/(4*(n1+n2) - 9)) if (n1+n2) > 2 else 1.0
    g  = d * J

    # 95% CI (Welch–Satterthwaite)
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

# -------- MAB helpers --------
RNG = np.random.default_rng(7)

def get_true_means(control, test, metric):
    mu_c = float(pd.to_numeric(control[metric], errors="coerce").mean(skipna=True))
    mu_t = float(pd.to_numeric(test[metric], errors="coerce").mean(skipna=True))
    return {"Control": mu_c, "Test": mu_t}

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

# ----------------------------
# 데이터 자동 로드
# ----------------------------
CTRL_PATH = Path("control.csv")
TEST_PATH = Path("test.csv")
if not CTRL_PATH.exists() or not TEST_PATH.exists():
    st.error("현재 폴더에서 control.csv / test.csv 를 찾지 못했습니다. 파일을 같은 폴더에 두고 다시 실행하세요.")
    st.stop()

control = prepare_df(pd.read_csv(CTRL_PATH))
test    = prepare_df(pd.read_csv(TEST_PATH))

# ----------------------------
# KPI (Revenue → ROAS → Frequency → CTR → CVR)
# ----------------------------
st.title("A/B Test Dashboard + MAB")

cols = st.columns(5)
for i, label in enumerate(["Revenue","ROAS","Frequency","CTR","CVR"]):
    if label in control.columns and label in test.columns:
        c = float(pd.to_numeric(control[label], errors="coerce").mean())
        t = float(pd.to_numeric(test[label], errors="coerce").mean())
        delta = (t - c) / c * 100 if c else np.nan
        cols[i].metric(label, f"{t:,.4g}", f"{delta:+.1f}% vs Control")

# ----------------------------
# 1) Welch t-test
# ----------------------------
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

# ----------------------------
# 2) 퍼널 분석
# ----------------------------
st.header("2) 퍼널 분석")

# ---- 2-A. 합계 퍼널 (Matplotlib 스타일) ----
st.subheader("2-A) Funnel (Stage counts): Impressions → Clicks → Purchases → Revenue")

def stage_totals(df):
    d = {}
    if "# of Impressions" in df.columns: d["Impressions"] = pd.to_numeric(df["# of Impressions"], errors="coerce").sum()
    if "# of Website Clicks" in df.columns: d["Clicks"] = pd.to_numeric(df["# of Website Clicks"], errors="coerce").sum()
    if "# of Purchase" in df.columns: d["Purchases"] = pd.to_numeric(df["# of Purchase"], errors="coerce").sum()
    if "Revenue" in df.columns: d["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce").sum()
    return d

ctrl_stage = stage_totals(control)
test_stage = stage_totals(test)
stages = ["Impressions", "Clicks", "Purchases", "Revenue"]

ctrl_vals = [ctrl_stage.get(s, 0) for s in stages]
test_vals = [test_stage.get(s, 0) for s in stages]

if MATPLOTLIB_OK:
    # --- 당신이 준 코드 그대로 Streamlit에 렌더 ---
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

    y_idx = np.arange(len(stages))[::-1]
    bar_h = 0.36
    offset = 0.18

    b1 = ax.barh(y_idx + offset, ctrl_vals[::-1], height=bar_h, label="Control")
    b2 = ax.barh(y_idx - offset, test_vals[::-1], height=bar_h, label="Test")

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    xmax = max(max(ctrl_vals), max(test_vals)) if (ctrl_vals and test_vals) else 0
    ax.set_xlim(0, xmax * 1.12 if xmax > 0 else 1)

    ax.set_yticks(y_idx)
    ax.set_yticklabels(stages[::-1])
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_xlabel("Counts / Revenue")
    ax.set_title("Funnel (Stage counts): Impressions → Clicks → Purchases → Revenue")

    def add_labels(bars, color="black", dx=0.01):
        for rect in bars:
            w = rect.get_width()
            ymid = rect.get_y() + rect.get_height()/2
            ax.text(w + (xmax*dx if xmax>0 else 0.02), ymid, f"{int(round(w)):,}",
                    va="center", ha="left", fontsize=10, color=color)

    add_labels(b1, color="black", dx=0.008)
    add_labels(b2, color="black", dx=0.008)

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), frameon=False)
    st.pyplot(fig)
else:
    # Matplotlib이 없을 때 Plotly로 유사 스타일 출력
    fig_f = go.Figure()
    fig_f.add_trace(go.Bar(y=stages, x=ctrl_vals, name="Control", orientation="h",
                           text=[f"{int(v):,}" for v in ctrl_vals], textposition="outside"))
    fig_f.add_trace(go.Bar(y=stages, x=test_vals, name="Test", orientation="h",
                           text=[f"{int(v):,}" for v in test_vals], textposition="outside"))
    xmax = max(max(ctrl_vals), max(test_vals)) if (ctrl_vals and test_vals) else 0
    fig_f.update_layout(
        barmode="group", bargap=0.2, height=450,
        xaxis=dict(title="Counts / $", tickformat=",d", range=[0, xmax*1.12 if xmax>0 else 1],
                   gridcolor="rgba(255,255,255,0.2)"),
        title="Funnel (Stage counts): Impressions → Clicks → Purchases → Revenue",
        legend=dict(orientation="v", x=1.02, y=1.0), margin=dict(r=140)
    )
    st.plotly_chart(fig_f, use_container_width=True)

# ---- 2-B. 퍼널 플로우 (평균 기반: 250827.py와 동일) ----
st.subheader("2-B) Funnel Flow (평균 기반, Revenue 1/10,000 정규화)")
steps = [s for s in ["CTR","CVR","Revenue"] if s in res_df["Metric"].values]
if steps:
    mean_ctrl = [float(res_df.loc[res_df['Metric']==s, "Control mean"].iloc[0]) for s in steps]
    mean_test = [float(res_df.loc[res_df['Metric']==s, "Test mean"].iloc[0])     for s in steps]
    if "Revenue" in steps:
        i = steps.index("Revenue")
        mean_ctrl[i] /= 10000.0
        mean_test[i] /= 10000.0

    def _lab(v, name): return f"{v:.3f}" if name!="Revenue" else f"{v*10000:.0f}"
    labels_ctrl = [_lab(v, s) for v, s in zip(mean_ctrl, steps)]
    labels_test = [_lab(v, s) for v, s in zip(mean_test, steps)]

    yy = list(range(len(steps)))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=mean_ctrl, y=yy, mode="lines+markers+text",
                              name="Control", text=labels_ctrl, textposition="top center",
                              line=dict(color="gray")))
    fig2.add_trace(go.Scatter(x=mean_test, y=yy, mode="lines+markers+text",
                              name="Test", text=labels_test, textposition="bottom center",
                              line=dict(color="royalblue")))
    fig2.update_yaxes(tickmode="array", tickvals=yy,
                      ticktext=[f"{s} (norm)" if s=="Revenue" else s for s in steps],
                      autorange="reversed")
    fig2.update_layout(height=340, title="Funnel Flow: CTR → CVR → Revenue",
                       xaxis_title="Relative scale (Revenue normalized by 10,000)",
                       legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98))
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# 3) 멀티암 밴딧 시뮬레이터
# ----------------------------
st.header("3) 멀티암 밴딧 시뮬레이터")
with st.expander("시뮬레이션 옵션", expanded=True):
    n_rounds = st.slider("Rounds", 200, 5000, 2000, 100)
    epsilon  = st.slider("ε (epsilon-greedy)", 0.00, 0.50, 0.10, 0.01)

tab_ctr, tab_cvr, tab_roas = st.tabs(["CTR (Clicks)", "CVR (Conversions)", "ROAS"])

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

with tab_cvr:
    if {"CVR"} <= set(control.columns) and {"CVR"} <= set(test.columns):
        true_cvr = get_true_means(control, test, "CVR")
        cvr_eps = simulate_epsilon_greedy(true_cvr, n_rounds, epsilon, bernoulli=True)
        cvr_ts  = simulate_thompson_bernoulli(true_cvr, n_rounds)
        cvr_ab  = simulate_ab_fixed(true_cvr, n_rounds, bernoulli=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=cvr_eps, x=np.arange(n_rounds), name="ε-greedy"))
        fig.add_trace(go.Scatter(y=cvr_ts,  x=np.arange(n_rounds), name="Thompson"))
        fig.add_trace(go.Scatter(y=cvr_ab,  x=np.arange(n_rounds), name="A/B (50:50)", line=dict(dash="dash")))
        fig.update_layout(height=380, title="Cumulative Conversions", xaxis_title="Round", yaxis_title="Cumulative")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Control CVR ≈ {true_cvr['Control']:.4f}, Test CVR ≈ {true_cvr['Test']:.4f}")

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

# ----------------------------
# 4) 시계열 비교
# ----------------------------
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

# ----------------------------
# 5) 결과 다운로드
# ----------------------------
csv_buf = io.StringIO()
res_df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
st.download_button("결과 CSV 다운로드", data=csv_buf.getvalue(),
                   file_name="ab_welch_results.csv", mime="text/csv")

st.success("✅ 준비 완료 — control.csv / test.csv 있는 폴더에서 실행하세요.")
