"""
MA343 – Coursework 1: Soccer Player Performance Analysis
Robust Python Scaffold (v3)
- 统一数值清洗（去逗号/货币符号/空白并强制转为数值）
- VIF 前强制数值化 + 去除常量列/零方差列，避免 np.isfinite 报错
- 拟合前对参与列做 dropna
Outputs -> ./outputs
"""

# ========= Config =========
DATA_PATH = "soccer_players_22.xlsx"

# ========= Imports =========
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.anova import anova_lm
from scipy.stats import skew

os.makedirs("outputs", exist_ok=True)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def read_any(path):
    low = path.lower()
    if low.endswith(".csv"):
        return pd.read_csv(path)
    if low.endswith(".xlsx") or low.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Use CSV/XLSX.")

# ========= Load & Clean =========
df = read_any(DATA_PATH)

# 标准化列名
df.columns = (
    df.columns
      .str.strip()
      .str.replace(" ", "_")
      .str.replace("-", "_")
      .str.replace("(", "", regex=False)
      .str.replace(")", "", regex=False)
      .str.lower()
)

# 可能是数值的列（按作业字段）
maybe_numeric = [
    "value_eur","overall","potential","age","height_cm","weight_kg",
    "pace","shooting","passing","dribbling","defending","physic"
]

# 先清理常见符号，再强制转为数值（无法转的设为 NaN）
for col in df.columns:
    if col in maybe_numeric or df[col].dtype == "object":
        s = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        # 只对 maybe_numeric 做强制数值化；其他保持原状
        if col in maybe_numeric:
            df[col] = pd.to_numeric(s, errors="coerce")
        else:
            df[col] = s

# 关键列缺失直接丢弃
df = df.dropna(subset=["value_eur", "overall", "potential"]).copy()

# ========== Task 1 ==========
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
summary = df[num_cols].agg(["mean","median","std","min","max","count"]).T
summary.to_csv("outputs/task1_summary_stats.csv")

for col in ["value_eur","overall","potential"]:
    s = skew(df[col].dropna())
    with open(f"outputs/task1_{col}_skew.txt","w",encoding="utf-8") as f:
        f.write(f"Skew({col}) = {s:.4f}\n")
        f.write("Interpretation: " + ("Right-skewed\n" if s>0.2 else "Left-skewed\n" if s<-0.2 else "Approximately symmetric\n"))
    plt.figure()
    df[col].hist(bins=40)
    plt.xlabel(col); plt.ylabel("Frequency"); plt.title(f"Histogram of {col}")
    savefig(f"outputs/task1_{col}_hist.png")

corr = df["value_eur"].corr(df["overall"])
with open("outputs/task1_corr_value_overall.txt","w") as f:
    f.write(f"Pearson r(value_eur, overall) = {corr:.4f}\n")

df[num_cols].corr().to_csv("outputs/task1_corr_matrix.csv")

if "club_position" in df.columns:
    tbl = (
        df.groupby("club_position")
          .agg(counts=("club_position","size"),
               mean_value_eur=("value_eur","mean"))
          .sort_values("mean_value_eur", ascending=False)
    )
    tbl.to_csv("outputs/task1_by_position.csv")
    plt.figure(figsize=(10,6))
    tbl["mean_value_eur"].plot(kind="bar")
    plt.ylabel("Mean value (EUR)"); plt.title("Mean Market Value by Club Position")
    savefig("outputs/task1_by_position_bar.png")

# ========== Task 2 ==========
model1 = smf.ols("value_eur ~ overall", data=df).fit()
model2 = smf.ols("value_eur ~ potential", data=df).fit()

best_predictor = "overall" if model1.rsquared > model2.rsquared else "potential"
best_model = model1 if best_predictor=="overall" else model2

with open("outputs/task2_models.txt","w") as f:
    f.write("Model A: value_eur ~ overall\n")
    f.write(model1.summary().as_text() + "\n\n")
    f.write("Model B: value_eur ~ potential\n")
    f.write(model2.summary().as_text() + "\n")

plt.figure()
plt.scatter(best_model.fittedvalues, best_model.resid, s=10)
plt.axhline(0, ls="--"); plt.xlabel("Fitted"); plt.ylabel("Residuals")
plt.title(f"Residuals vs Fitted ({best_predictor})")
savefig("outputs/task2_resid_vs_fitted_simple.png")

plt.figure()
qqplot(best_model.resid, line='45')
plt.title("QQ-plot (Simple Model)")
savefig("outputs/task2_qq_simple.png")

# 二次 & 对数模型
X = best_predictor
df[f"{X}_sq"] = df[X]**2
quad = smf.ols(f"value_eur ~ {X} + {X}_sq", data=df).fit()
with open("outputs/task2_quad.txt","w") as f:
    f.write(quad.summary().as_text())

df["ln_value_eur"] = np.log(df["value_eur"].clip(lower=1))
logm = smf.ols(f"ln_value_eur ~ {X} + {X}_sq", data=df).fit()
with open("outputs/task2_log_model.txt","w") as f:
    f.write(logm.summary().as_text())

plt.figure()
plt.scatter(logm.fittedvalues, logm.resid, s=10)
plt.axhline(0, ls="--"); plt.xlabel("Fitted"); plt.ylabel("Residuals")
plt.title("Residuals vs Fitted (Log Model)")
savefig("outputs/task2_resid_vs_fitted_log.png")

# 影响点（Cook）
n = int(logm.nobs); p = int(logm.df_model) + 1
threshold = 4 / max(1, (n - p - 1))
cooks = OLSInfluence(logm).cooks_distance[0]
influential_idx = np.where(cooks > threshold)[0].tolist()
with open("outputs/task2_influential.txt","w") as f:
    f.write(f"Threshold = {threshold:.6f}\nInfluential points: {len(influential_idx)}\n")

# ========== Task 3 ==========
ATTRS = ["age","height_cm","weight_kg","pace","shooting","passing","dribbling","defending","physic"]

def _prep_numeric(df_in, cols):
    """强制把指定列转为 float，并丢掉含 NaN 的行；去除零方差列。"""
    sub = df_in[cols].apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna(axis=0, how="any").copy()
    # 去掉零方差列，避免 VIF/回归数值问题
    var0 = sub.std(axis=0) == 0
    if var0.any():
        sub = sub.loc[:, ~var0]
    return sub

def run_vif_loop(df_in, y_col):
    current_attrs = [c for c in ATTRS if c in df_in.columns]
    removed = []
    while True:
        Xdf = _prep_numeric(df_in, current_attrs)
        if Xdf.empty or Xdf.shape[1] <= 1:
            break
        Xc = sm.add_constant(Xdf, has_constant="add")
        # 计算 VIF
        vif_vals = []
        for i in range(1, Xc.shape[1]):  # 跳过 const
            vif_vals.append(variance_inflation_factor(Xc.values, i))
        vif = pd.Series(vif_vals, index=Xdf.columns, name="VIF")
        vif.to_csv(f"outputs/task3_vif_{y_col}.csv")
        max_var, max_v = vif.idxmax(), float(vif.max())
        if max_v > 5 and len(current_attrs) > 1:
            removed.append(max_var)
            current_attrs.remove(max_var)
        else:
            break
    return current_attrs, removed

def fit_and_save(df_in, y_col, attrs):
    # 1) 先把 y 和 X 合在一起，做统一的数值清洗 & 丢缺失 & 去零方差
    cols = [y_col] + attrs
    sub = df_in[cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any").copy()
    # 去掉零方差列（除 y 以外）
    var0 = sub.drop(columns=[y_col]).std(axis=0) == 0
    if var0.any():
        sub = sub.loc[:, [y_col] + var0.index[~var0].tolist()]

    # 2) 用“公式接口”拟合（这会自动携带 design_info，anova_lm 才能用）
    rhs = " + ".join([c for c in sub.columns if c != y_col])
    formula = f"{y_col} ~ {rhs}"
    model = smf.ols(formula, data=sub).fit()

    # 3) 导出回归结果 & ANOVA（Type II）
    with open(f"outputs/task3_{y_col}_model.txt","w", encoding="utf-8") as f:
        f.write(model.summary().as_text())
    anova_lm(model, typ=2).to_csv(f"outputs/task3_{y_col}_anova.csv")

    # 4) 残差图 & QQ 图
    plt.figure()
    plt.scatter(model.fittedvalues, model.resid, s=10)
    plt.axhline(0, ls="--"); plt.xlabel("Fitted"); plt.ylabel("Residuals")
    plt.title(f"Residuals vs Fitted ({y_col})")
    savefig(f"outputs/task3_{y_col}_resid_vs_fitted.png")

    plt.figure()
    qqplot(model.resid, line='45')
    plt.title(f"QQ-plot ({y_col})")
    savefig(f"outputs/task3_{y_col}_qq.png")

    return model


attrs_overall, removed_overall = run_vif_loop(df, "overall")
attrs_potential, removed_potential = run_vif_loop(df, "potential")

model_overall = fit_and_save(df, "overall", attrs_overall)
model_potential = fit_and_save(df, "potential", attrs_potential)

with open("outputs/task3_vif_summary.txt","w") as f:
    f.write("VIF loop (threshold=5)\n")
    f.write(f"overall kept={attrs_overall}, removed={removed_overall}\n")
    f.write(f"potential kept={attrs_potential}, removed={removed_potential}\n")

print("✅ Done. Check the 'outputs/' folder for figures/tables.")
# ============================
# === Task 4 — More Models (Bonus 6 pt)
# ============================
# 目标：
# (1) 检验 preferred_foot 是否能预测 overall
# (2) 用 overall（数值）和 club_position（类别）以及两者交互项来预测 value_eur
# (3) 若发现明显假设违背，给出调整（稳健标准误 & 结果解释）

from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan

# ---------- T4(1): overall ~ preferred_foot ----------
if "preferred_foot" in df.columns:
    sub_pf = df[[ "overall", "preferred_foot" ]].dropna().copy()
    # 保证是类别
    sub_pf["preferred_foot"] = sub_pf["preferred_foot"].astype("category")

    # 公式法（带 design_info，ANOVA 可用）
    m_pf = smf.ols("overall ~ C(preferred_foot)", data=sub_pf).fit()
    with open("outputs/task4_overall_by_preferred_foot.txt","w", encoding="utf-8") as f:
        f.write(m_pf.summary().as_text())

    # Type II ANOVA（主效应显著性）
    anova_lm(m_pf, typ=2).to_csv("outputs/task4_overall_by_preferred_foot_anova.csv")

    # 组均值导出，便于直观比较（左脚 vs 右脚）
    grp = sub_pf.groupby("preferred_foot")["overall"].agg(["count","mean","std"]).sort_index()
    grp.to_csv("outputs/task4_overall_by_preferred_foot_group_stats.csv")

    # 残差诊断
    plt.figure()
    plt.scatter(m_pf.fittedvalues, m_pf.resid, s=10)
    plt.axhline(0, ls="--")
    plt.xlabel("Fitted"); plt.ylabel("Residuals")
    plt.title("Task4(1) Residuals vs Fitted: overall ~ preferred_foot")
    savefig("outputs/task4_pf_resid_vs_fitted.png")

    plt.figure()
    qqplot(m_pf.resid, line='45')
    plt.title("Task4(1) QQ-plot: overall ~ preferred_foot")
    savefig("outputs/task4_pf_qq.png")

    # 通俗解释
    with open("outputs/task4_overall_by_preferred_foot_plain.txt","w", encoding="utf-8") as f:
        f.write("Task 4(1) — Does preferred_foot predict overall?\n")
        f.write("- We compared mean overall between dominant-foot groups using a regression with a categorical predictor.\n")
        f.write("- Check the ANOVA CSV for whether footedness has a statistically detectable effect on overall.\n")
        f.write("- Group means are in task4_overall_by_preferred_foot_group_stats.csv for practical interpretation.\n")

# ---------- T4(2): value_eur ~ overall * club_position ----------
if "club_position" in df.columns:
    # 仅保留较常见的位置，避免过多稀疏类别引起不稳；其余合并为 'Other'
    pos_counts = df["club_position"].value_counts(dropna=True)
    top_positions = pos_counts.index[:8].tolist()  # 取前 8 个最常见位置
    df_pos = df[["value_eur","overall","club_position"]].dropna().copy()
    df_pos["club_position"] = df_pos["club_position"].where(
        df_pos["club_position"].isin(top_positions), other="Other"
    ).astype("category")

    # 对 overall 做中心化，便于交互解释且稳定数值
    df_pos["overall_c"] = df_pos["overall"] - df_pos["overall"].mean()

    # 基础 OLS（常规标准误）
    formula_int = "value_eur ~ overall_c * C(club_position)"
    m_int = smf.ols(formula_int, data=df_pos).fit()
    with open("outputs/task4_value_by_overallXposition.txt","w", encoding="utf-8") as f:
        f.write(m_int.summary().as_text())

    # Breusch–Pagan 异方差检验；若显著，则再给一版稳健标准误（HC3）
    bp_lm, bp_pval, _, _ = het_breuschpagan(m_int.resid, m_int.model.exog)
    with open("outputs/task4_value_by_overallXposition_bp.txt","w", encoding="utf-8") as f:
        f.write(f"Breusch-Pagan LM={bp_lm:.4f}, p-value={bp_pval:.4g}\n")
        if bp_pval < 0.05:
            f.write("Evidence of heteroskedasticity; see robust (HC3) summary.\n")
        else:
            f.write("No strong evidence of heteroskedasticity.\n")

    # 稳健标准误（HC3）
    m_int_hc3 = smf.ols(formula_int, data=df_pos).fit(cov_type="HC3")
    with open("outputs/task4_value_by_overallXposition_HC3.txt","w", encoding="utf-8") as f:
        f.write(m_int_hc3.summary().as_text())

    # 残差诊断图（用稳健 or 非稳健的残差都行；这里沿用 OLS）
    plt.figure()
    plt.scatter(m_int.fittedvalues, m_int.resid, s=10)
    plt.axhline(0, ls="--")
    plt.xlabel("Fitted"); plt.ylabel("Residuals")
    plt.title("Task4(2) Residuals vs Fitted: value_eur ~ overall_c * position")
    savefig("outputs/task4_interaction_resid_vs_fitted.png")

    plt.figure()
    qqplot(m_int.resid, line='45')
    plt.title("Task4(2) QQ-plot: value_eur ~ overall_c * position")
    savefig("outputs/task4_interaction_qq.png")

    # 便于快速理解交互：导出每个位置的线性斜率（overall 的边际效应）
    # 斜率 = overall_c 系数 +（overall_c:position=k 的交互系数）
    base_slope = m_int.params.get("overall_c", np.nan)
    slope_by_pos = {}
    for cat in df_pos["club_position"].cat.categories:
        if cat == df_pos["club_position"].cat.categories[0]:
            # 参考基类（由 patsy 自动选定，通常是按字母序的第一个）
            slope_by_pos[cat] = base_slope
        else:
            key = f"overall_c:C(club_position)[T.{cat}]"
            slope_by_pos[cat] = base_slope + m_int.params.get(key, 0.0)
    pd.Series(slope_by_pos, name="slope_of_overall").to_csv(
        "outputs/task4_interaction_slope_by_position.csv"
    )

    # 通俗解释
    with open("outputs/task4_value_by_overallXposition_plain.txt","w", encoding="utf-8") as f:
        f.write("Task 4(2) — Do position groups change the relationship between rating and market value?\n")
        f.write("- We fit an interaction model: value_eur ~ centered overall * club_position (top 8 positions + Other).\n")
        f.write("- If interaction terms are significant, it means the effect (slope) of overall on value varies by position.\n")
        f.write("- Check '..._HC3.txt' for robust standard errors in case of heteroskedasticity.\n")
        f.write("- File 'task4_interaction_slope_by_position.csv' lists the estimated slope for each position group.\n")

