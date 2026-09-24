"""
============================================================
  E-Commerce Sales & Customer Analytics
  IBM SkillsBuild Data Analytics with AI – Internship Project
============================================================

Business Intelligence Flow:
  Data → Information → Insights → Business Decisions → Recommended Actions

Dataset : ecommerce_sales_analytics_5000.csv  (synthetic / simulated)
Author  : (your name)
Date    : 2025

IMPORTANT – DATASET LIMITATION:
  This dataset is entirely synthetic/simulated. It contains future-dated
  orders (up to 2035), exactly one order per calendar day, and artificially
  uniform distributions. All findings are illustrative of analytical methods
  and should NOT be presented as real-world business facts.
============================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 │ Imports, Configuration, Output Setup
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Output folders ────────────────────────────────────────────────────────────
OUTPUT_DIR   = "output"
PAGE1_DIR    = os.path.join(OUTPUT_DIR, "page1_executive_overview")
PAGE2_DIR    = os.path.join(OUTPUT_DIR, "page2_sales_product")
PAGE3_DIR    = os.path.join(OUTPUT_DIR, "page3_customer_operations")

for d in [PAGE1_DIR, PAGE2_DIR, PAGE3_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Visual style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE   = ["#2563EB", "#7C3AED", "#059669", "#D97706", "#DC2626", "#0891B2"]
BRAND_CLR = "#2563EB"

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
})

DATA_FILE = "ecommerce_sales_analytics_5000.csv"

SEP  = "=" * 70
SEP2 = "-" * 70

def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")

def subsection(title: str) -> None:
    print(f"\n{SEP2}\n  {title}\n{SEP2}")

def save_fig(path: str, fig: plt.Figure) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  [saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: currency y-axis formatters (used in both analysis and Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def fmt_millions(x, _):
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    elif x >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:.0f}"

def fmt_currency(x, _):
    return f"${x:,.0f}"

NOTE = "Note: Synthetic dataset — for illustrative purposes only."

def add_note(ax):
    ax.annotate(NOTE, xy=(0, -0.13), xycoords="axes fraction",
                fontsize=7.5, color="#6B7280", style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis functions (shared by CLI run and Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare_data():
    """Load CSV, clean, transform, and return enriched DataFrame."""
    df = pd.read_csv(DATA_FILE)

    # Parse dates
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["year"]       = df["order_date"].dt.year
    df["month"]      = df["order_date"].dt.month
    df["year_month"] = df["order_date"].dt.to_period("M")

    # Discount bands
    disc_bins   = [-0.001, 0.10, 0.20, 0.35]
    disc_labels = ["Low (0–10%)", "Medium (11–20%)", "High (21–35%)"]
    df["discount_band"] = pd.cut(df["discount"], bins=disc_bins, labels=disc_labels)

    # Delivery bands
    del_bins   = [0, 3, 7, 11]
    del_labels = ["Fast (1–3 days)", "Normal (4–7 days)", "Slow (8–11 days)"]
    df["delivery_band"] = pd.cut(df["delivery_days"], bins=del_bins, labels=del_labels)

    # Rating bands
    rat_bins   = [0, 2, 3, 5]
    rat_labels = ["Poor (1–2)", "Neutral (3)", "Good (4–5)"]
    df["rating_band"] = pd.cut(df["customer_rating"], bins=rat_bins, labels=rat_labels)

    return df


def compute_kpis(df):
    """Compute all KPIs and aggregations from the DataFrame."""
    kpis = {}

    # Level 1 – What is happening?
    kpis["total_revenue"]      = df["revenue"].sum()
    kpis["total_orders"]       = len(df)
    kpis["total_customers"]    = df["customer_id"].nunique()
    kpis["aov"]                = kpis["total_revenue"] / kpis["total_orders"]
    kpis["avg_rating"]         = df["customer_rating"].mean()
    kpis["avg_delivery_days"]  = df["delivery_days"].mean()

    # Trends
    monthly_revenue = df.groupby("year_month")["revenue"].sum().reset_index()
    monthly_revenue["year_month_str"] = monthly_revenue["year_month"].astype(str)
    kpis["monthly_revenue"] = monthly_revenue
    kpis["annual_revenue"]  = df.groupby("year")["revenue"].sum()

    # Level 3 – Why?
    kpis["rev_by_category"]   = df.groupby("product_category")["revenue"].sum().sort_values(ascending=False)
    kpis["rev_by_region"]     = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    kpis["aov_by_category"]   = df.groupby("product_category")["revenue"].mean().sort_values(ascending=False)
    kpis["avg_rating_by_cat"] = df.groupby("product_category")["customer_rating"].mean().sort_values(ascending=False)
    kpis["avg_rating_by_reg"] = df.groupby("region")["customer_rating"].mean().sort_values(ascending=False)
    kpis["avg_del_by_region"] = df.groupby("region")["delivery_days"].mean().sort_values()
    kpis["rev_by_disc_band"]  = df.groupby("discount_band", observed=True)["revenue"].mean()
    kpis["rat_by_disc_band"]  = df.groupby("discount_band", observed=True)["customer_rating"].mean()
    kpis["rat_by_del"]        = df.groupby("delivery_band", observed=True)["customer_rating"].mean()
    kpis["aov_by_pm"]         = df.groupby("payment_method")["revenue"].mean().sort_values(ascending=False)
    kpis["pm_counts"]         = df["payment_method"].value_counts()

    # Level 4 – Risks & Opportunities
    cust_orders             = df.groupby("customer_id")["order_id"].count()
    kpis["cust_orders"]     = cust_orders
    repeat_customers        = (cust_orders > 1).sum()
    kpis["repeat_customers"] = repeat_customers
    kpis["repeat_rate"]      = repeat_customers / kpis["total_customers"] * 100

    cust_revenue             = df.groupby("customer_id")["revenue"].sum().sort_values(ascending=False)
    kpis["cust_revenue"]     = cust_revenue
    top10pct_n               = max(1, int(kpis["total_customers"] * 0.10))
    kpis["top10pct_n"]       = top10pct_n
    kpis["top10pct_share"]   = cust_revenue.head(top10pct_n).sum() / kpis["total_revenue"] * 100

    kpis["poor_rating_n"]     = (df["customer_rating"] <= 2).sum()
    kpis["poor_rating_rate"]  = kpis["poor_rating_n"] / kpis["total_orders"] * 100
    kpis["slow_delivery_n"]   = (df["delivery_days"] >= 9).sum()
    kpis["slow_delivery_rate"] = kpis["slow_delivery_n"] / kpis["total_orders"] * 100
    kpis["high_disc_n"]       = (df["discount"] >= 0.25).sum()
    kpis["high_disc_rate"]    = kpis["high_disc_n"] / kpis["total_orders"] * 100

    return kpis


def build_insights(kpis):
    """Return the list of structured business insights."""
    rev_by_category   = kpis["rev_by_category"]
    rev_by_region     = kpis["rev_by_region"]
    avg_rating_by_cat = kpis["avg_rating_by_cat"]
    avg_rating_by_reg = kpis["avg_rating_by_reg"]
    avg_del_by_region = kpis["avg_del_by_region"]
    rev_by_disc_band  = kpis["rev_by_disc_band"]
    rat_by_disc_band  = kpis["rat_by_disc_band"]
    rat_by_del        = kpis["rat_by_del"]
    aov_by_pm         = kpis["aov_by_pm"]
    total_revenue     = kpis["total_revenue"]
    repeat_rate       = kpis["repeat_rate"]
    repeat_customers  = kpis["repeat_customers"]
    total_customers   = kpis["total_customers"]
    top10pct_share    = kpis["top10pct_share"]
    top10pct_n        = kpis["top10pct_n"]
    poor_rating_rate  = kpis["poor_rating_rate"]
    poor_rating_n     = kpis["poor_rating_n"]

    top_category     = rev_by_category.idxmax()
    top_category_pct = rev_by_category.max() / total_revenue * 100
    top_region       = rev_by_region.idxmax()
    top_region_pct   = rev_by_region.max() / total_revenue * 100
    worst_cat_rating = avg_rating_by_cat.idxmin()
    worst_reg_rating = avg_rating_by_reg.idxmin()
    disc_low_rev     = rev_by_disc_band.get("Low (0–10%)", 0)
    disc_high_rev    = rev_by_disc_band.get("High (21–35%)", 0)
    disc_low_rat     = rat_by_disc_band.get("Low (0–10%)", 0)
    disc_high_rat    = rat_by_disc_band.get("High (21–35%)", 0)
    rat_fast         = rat_by_del.get("Fast (1–3 days)", 0)
    rat_slow         = rat_by_del.get("Slow (8–11 days)", 0)

    return [
        {
            "id": "INS-01",
            "topic": "Revenue Concentration by Category",
            "observation": (f"{top_category} is the top revenue category, contributing {top_category_pct:.1f}% of total revenue."),
            "insight": ("One product category dominates revenue, creating concentration risk. The remaining categories generate proportionally lower revenue despite similar order volumes."),
            "implication": ("Over-reliance on a single category exposes the business to demand fluctuations in that segment."),
            "action": ("Invest in marketing and inventory for under-performing categories to diversify the revenue base."),
        },
        {
            "id": "INS-02",
            "topic": "Regional Revenue Distribution",
            "observation": (f"{top_region} leads on revenue with {top_region_pct:.1f}% of total; all regions are relatively balanced."),
            "insight": ("Revenue is nearly evenly distributed across regions, suggesting no single geography is a dominant growth driver."),
            "implication": ("Equal regional distribution limits the ability to identify a high-priority expansion market from revenue data alone."),
            "action": ("Cross-reference revenue with order count and customer rating per region to identify high-potential regions worth further investment."),
        },
        {
            "id": "INS-03",
            "topic": "Discount Strategy Effectiveness",
            "observation": (f"High-discount orders (21–35%) produce avg revenue ${disc_high_rev:,.0f} vs ${disc_low_rev:,.0f} for low-discount orders. High-discount avg rating: {disc_high_rat:.2f} vs {disc_low_rat:.2f} for low."),
            "insight": ("Higher discounts generate higher absolute revenue per order (because discount applies to a larger basket), but do NOT meaningfully improve customer satisfaction ratings."),
            "implication": ("Discounting is likely margin-erosive: it increases order size without generating a corresponding loyalty or satisfaction benefit."),
            "action": ("Cap the maximum discount. Redirect promotional budget to delivery speed improvements or product quality, which have a stronger link to customer satisfaction."),
        },
        {
            "id": "INS-04",
            "topic": "Delivery Speed & Customer Satisfaction",
            "observation": (f"Fast deliveries (1–3 days) avg rating: {rat_fast:.2f}. Slow deliveries (8–11 days) avg rating: {rat_slow:.2f}."),
            "insight": ("Delivery speed is the most direct operational lever for customer satisfaction. Slower deliveries consistently produce lower customer ratings."),
            "implication": ("Every day added to delivery time reduces customer satisfaction, increasing churn risk and reducing repeat purchase likelihood."),
            "action": (f"Set a target maximum delivery time of 7 days. Prioritise logistics investment in regions with the highest average delivery days ({avg_del_by_region.idxmax()}: {avg_del_by_region.max():.1f} days)."),
        },
        {
            "id": "INS-05",
            "topic": "Customer Loyalty & Repeat Purchase Rate",
            "observation": (f"{repeat_rate:.1f}% of customers ({repeat_customers:,} of {total_customers:,}) placed more than one order. Top 10% of customers account for {top10pct_share:.1f}% of revenue."),
            "insight": ("The customer base is highly loyal — nearly all customers return for repeat purchases. However, a small top segment generates a disproportionate share of revenue."),
            "implication": ("Revenue is heavily dependent on a small, loyal customer pool. Losing these customers would have an outsized revenue impact."),
            "action": ("Implement a tiered loyalty programme. Identify and proactively engage the top revenue customers with personalised offers, early access, or dedicated support."),
        },
        {
            "id": "INS-06",
            "topic": "Customer Satisfaction Weak Spots",
            "observation": (f"{worst_cat_rating} has the lowest avg category rating; {worst_reg_rating} has the lowest avg regional rating. Overall poor rating rate (≤2): {poor_rating_rate:.1f}%."),
            "insight": ("Satisfaction issues are concentrated in specific category-region combinations, not uniform across the business."),
            "implication": ("A one-size-fits-all improvement programme will miss the root cause. Localised issues require targeted fixes."),
            "action": (f"Conduct root-cause analysis for {worst_cat_rating} category and {worst_reg_rating} region. Investigate whether delivery time, product quality, or pricing is the primary dissatisfier."),
        },
        {
            "id": "INS-07",
            "topic": "Payment Method & Order Value",
            "observation": (f"Card payments have the highest AOV (${aov_by_pm.get('Card', 0):,.0f}); Wallet has the lowest (${aov_by_pm.get('Wallet', 0):,.0f})."),
            "insight": ("Customers paying by card tend to place higher-value orders, while digital wallet users place lower-value orders on average."),
            "implication": ("Payment method can serve as a proxy for customer spend tier, enabling targeted promotions."),
            "action": ("Offer higher-value promotions to card users. Consider wallet-specific incentives (e.g. cashback) to encourage wallet users to increase basket size."),
        },
    ]


def generate_charts(df, kpis):
    """Generate and save all 15 charts to the output folders."""

    monthly_revenue   = kpis["monthly_revenue"]
    rev_by_category   = kpis["rev_by_category"]
    rev_by_region     = kpis["rev_by_region"]
    rev_by_disc_band  = kpis["rev_by_disc_band"]
    rat_by_disc_band  = kpis["rat_by_disc_band"]
    avg_del_by_region = kpis["avg_del_by_region"]
    cust_orders       = kpis["cust_orders"]
    cust_revenue      = kpis["cust_revenue"]
    rat_by_del        = kpis["rat_by_del"]
    aov_by_pm         = kpis["aov_by_pm"]
    pm_counts         = kpis["pm_counts"]

    # ── Chart 1 │ Monthly Revenue Trend ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly_revenue["year_month_str"], monthly_revenue["revenue"],
            color=BRAND_CLR, linewidth=1.8, alpha=0.9)
    ax.fill_between(monthly_revenue["year_month_str"], monthly_revenue["revenue"],
                    alpha=0.12, color=BRAND_CLR)
    years_idx = monthly_revenue[monthly_revenue["year_month_str"].str.endswith("-01")].index
    ax.set_xticks(years_idx)
    ax.set_xticklabels(
        [monthly_revenue.loc[i, "year_month_str"][:4] for i in years_idx],
        rotation=45, ha="right"
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
    ax.set_title("Monthly Revenue Trend (2022–2035)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Revenue")
    add_note(ax)
    save_fig(os.path.join(PAGE1_DIR, "01_monthly_revenue_trend.png"), fig)

    # ── Chart 2 │ Revenue by Category ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    cats  = rev_by_category.index.tolist()
    vals  = rev_by_category.values
    bars  = ax.bar(cats, vals, color=PALETTE[:len(cats)], edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8000,
                f"${val/1e6:.2f}M", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
    ax.set_title("Total Revenue by Product Category")
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Total Revenue")
    add_note(ax)
    save_fig(os.path.join(PAGE1_DIR, "02_revenue_by_category.png"), fig)

    # ── Chart 3 │ Revenue by Region ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    regs  = rev_by_region.index.tolist()
    rvals = rev_by_region.values
    hbars = ax.barh(regs, rvals, color=PALETTE[:len(regs)], edgecolor="white", linewidth=0.8)
    for bar, val in zip(hbars, rvals):
        ax.text(val + 5000, bar.get_y() + bar.get_height() / 2,
                f"${val/1e6:.2f}M", va="center", fontsize=9.5, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
    ax.set_title("Total Revenue by Region")
    ax.set_xlabel("Total Revenue")
    ax.set_ylabel("Region")
    ax.invert_yaxis()
    add_note(ax)
    save_fig(os.path.join(PAGE1_DIR, "03_revenue_by_region.png"), fig)

    # ── Chart 4 │ Orders by Payment Method ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        pm_counts.values,
        labels=pm_counts.index,
        autopct="%1.1f%%",
        colors=PALETTE[:len(pm_counts)],
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        pctdistance=0.78,
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    ax.set_title("Orders by Payment Method")
    ax.annotate(NOTE, xy=(0.5, -0.04), xycoords="axes fraction",
                ha="center", fontsize=7.5, color="#6B7280", style="italic")
    save_fig(os.path.join(PAGE1_DIR, "04_orders_by_payment_method.png"), fig)

    # ── Chart 5 │ Avg Revenue by Discount Band ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    disc_labels_plot = [str(x) for x in rev_by_disc_band.index]
    disc_vals        = rev_by_disc_band.values
    dbars = ax.bar(disc_labels_plot, disc_vals, color=PALETTE[:3], edgecolor="white")
    for bar, val in zip(dbars, disc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"${val:,.0f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
    ax.set_title("Average Order Revenue by Discount Band")
    ax.set_xlabel("Discount Band")
    ax.set_ylabel("Average Revenue per Order")
    add_note(ax)
    save_fig(os.path.join(PAGE2_DIR, "05_avg_revenue_by_discount_band.png"), fig)

    # ── Chart 6 │ Avg Rating by Discount Band ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    rat_vals = rat_by_disc_band.values
    rbars = ax.bar(disc_labels_plot, rat_vals, color=PALETTE[:3], edgecolor="white")
    for bar, val in zip(rbars, rat_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 5)
    ax.set_title("Average Customer Rating by Discount Band")
    ax.set_xlabel("Discount Band")
    ax.set_ylabel("Average Customer Rating (out of 5)")
    add_note(ax)
    save_fig(os.path.join(PAGE2_DIR, "06_avg_rating_by_discount_band.png"), fig)

    # ── Chart 7 │ AOV by Category × Region heatmap ───────────────────────────
    aov_pivot = df.pivot_table(
        index="product_category", columns="region",
        values="revenue", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        aov_pivot, annot=True, fmt=".0f", cmap="Blues",
        linewidths=0.5, linecolor="#e5e7eb",
        cbar_kws={"label": "Avg Revenue ($)"},
        ax=ax
    )
    ax.set_title("Average Order Value (AOV) — Category × Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Product Category")
    add_note(ax)
    save_fig(os.path.join(PAGE2_DIR, "07_aov_category_x_region_heatmap.png"), fig)

    # ── Chart 8 │ Category Revenue Share over Time ────────────────────────────
    cat_monthly = (
        df.groupby(["year", "product_category"])["revenue"]
        .sum()
        .reset_index()
        .pivot(index="year", columns="product_category", values="revenue")
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        cat_monthly.index,
        [cat_monthly[c] for c in cat_monthly.columns],
        labels=cat_monthly.columns.tolist(),
        colors=PALETTE[:len(cat_monthly.columns)],
        alpha=0.85
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
    ax.set_title("Annual Revenue by Category (Stacked)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Revenue")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
    add_note(ax)
    save_fig(os.path.join(PAGE2_DIR, "08_category_revenue_stacked_area.png"), fig)

    # ── Chart 9 │ Unit Price Distribution by Category ─────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    category_order = df.groupby("product_category")["unit_price"].median().sort_values(ascending=False).index.tolist()
    sns.boxplot(
        data=df, x="product_category", y="unit_price",
        order=category_order,
        palette=PALETTE[:4], ax=ax,
        linewidth=1.2, flierprops={"marker": "o", "markersize": 3, "alpha": 0.5}
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
    ax.set_title("Unit Price Distribution by Product Category")
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Unit Price ($)")
    add_note(ax)
    save_fig(os.path.join(PAGE2_DIR, "09_unit_price_distribution_by_category.png"), fig)

    # ── Chart 10 │ Customer Order Frequency ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    freq_counts = cust_orders.value_counts().sort_index()
    ax.bar(freq_counts.index, freq_counts.values, color=BRAND_CLR, edgecolor="white", linewidth=0.8)
    ax.set_title("Customer Order Frequency Distribution")
    ax.set_xlabel("Number of Orders per Customer")
    ax.set_ylabel("Number of Customers")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    add_note(ax)
    save_fig(os.path.join(PAGE3_DIR, "10_customer_order_frequency.png"), fig)

    # ── Chart 11 │ Top 20 Customers by Total Revenue ──────────────────────────
    top20 = cust_revenue.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        [f"Cust {cid}" for cid in top20.index],
        top20.values,
        color=BRAND_CLR, edgecolor="white", linewidth=0.8
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
    ax.set_title("Top 20 Customers by Total Revenue")
    ax.set_xlabel("Total Revenue ($)")
    ax.set_ylabel("Customer ID")
    ax.invert_yaxis()
    add_note(ax)
    save_fig(os.path.join(PAGE3_DIR, "11_top20_customers_by_revenue.png"), fig)

    # ── Chart 12 │ Avg Rating by Delivery Band ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    del_labels_plot = [str(x) for x in rat_by_del.index]
    delbars = ax.bar(del_labels_plot, rat_by_del.values,
                     color=["#059669", "#2563EB", "#DC2626"], edgecolor="white")
    for bar, val in zip(delbars, rat_by_del.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 5)
    ax.set_title("Average Customer Rating by Delivery Speed")
    ax.set_xlabel("Delivery Band")
    ax.set_ylabel("Average Customer Rating (out of 5)")
    add_note(ax)
    save_fig(os.path.join(PAGE3_DIR, "12_avg_rating_by_delivery_band.png"), fig)

    # ── Chart 13 │ Avg Delivery Days by Region ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    del_regs  = avg_del_by_region.index.tolist()
    del_rvals = avg_del_by_region.values
    del_rbars = ax.bar(del_regs, del_rvals, color=PALETTE[:len(del_regs)], edgecolor="white")
    for bar, val in zip(del_rbars, del_rvals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}d", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, max(del_rvals) * 1.15)
    ax.set_title("Average Delivery Days by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Average Delivery Days")
    add_note(ax)
    save_fig(os.path.join(PAGE3_DIR, "13_avg_delivery_days_by_region.png"), fig)

    # ── Chart 14 │ Avg Rating by Category × Region heatmap ───────────────────
    rating_pivot = df.pivot_table(
        index="product_category", columns="region",
        values="customer_rating", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        rating_pivot, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=1, vmax=5, linewidths=0.5, linecolor="#e5e7eb",
        cbar_kws={"label": "Avg Rating (1–5)"},
        ax=ax
    )
    ax.set_title("Average Customer Rating — Category × Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Product Category")
    add_note(ax)
    save_fig(os.path.join(PAGE3_DIR, "14_avg_rating_category_x_region_heatmap.png"), fig)

    # ── Chart 15 │ AOV by Payment Method ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    pmbars = ax.bar(aov_by_pm.index, aov_by_pm.values,
                    color=PALETTE[:len(aov_by_pm)], edgecolor="white")
    for bar, val in zip(pmbars, aov_by_pm.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"${val:,.0f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
    ax.set_title("Average Order Value (AOV) by Payment Method")
    ax.set_xlabel("Payment Method")
    ax.set_ylabel("Average Order Value ($)")
    add_note(ax)
    save_fig(os.path.join(PAGE3_DIR, "15_aov_by_payment_method.png"), fig)

    print(f"\n  ✓ All 15 charts saved to '{OUTPUT_DIR}/'")


# ─────────────────────────────────────────────────────────────────────────────
# CLI (non-Streamlit) execution path
# Runs when: python ecommerce_analytics.py
# ─────────────────────────────────────────────────────────────────────────────

def run_cli():
    """Execute the full CLI analysis pipeline with printed output."""

    # ── Section 1 │ Data Loading & Validation ─────────────────────────────────
    section("SECTION 1 │ DATA LOADING & VALIDATION")

    df_raw = pd.read_csv(DATA_FILE)

    print(f"\n  Rows    : {len(df_raw):,}")
    print(f"  Columns : {df_raw.shape[1]}")
    print(f"\n  Column names : {df_raw.columns.tolist()}")

    subsection("Data Types")
    print(df_raw.dtypes.to_string())

    subsection("Missing Values")
    missing = df_raw.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values found in any column.")
    else:
        print(missing[missing > 0])

    subsection("Duplicate Records")
    dup_rows     = df_raw.duplicated().sum()
    dup_orderids = df_raw["order_id"].duplicated().sum()
    print(f"  Fully duplicate rows : {dup_rows}")
    print(f"  Duplicate order_ids  : {dup_orderids}")
    if dup_rows == 0:
        print("  ✓ Dataset is clean – no duplicates detected.")

    subsection("Sample Rows")
    print(df_raw.head(3).to_string(index=False))

    # ── Section 2 │ Data Cleaning & Transformation ────────────────────────────
    section("SECTION 2 │ DATA CLEANING & TRANSFORMATION")

    df = load_and_prepare_data()

    df_raw["order_date"] = pd.to_datetime(df_raw["order_date"])
    print(f"\n  Date range : {df_raw['order_date'].min().date()}  →  {df_raw['order_date'].max().date()}")

    df_raw["revenue_check"] = (df_raw["quantity"] * df_raw["unit_price"] * (1 - df_raw["discount"])).round(2)
    mismatch = (df_raw["revenue"].round(2) != df_raw["revenue_check"]).sum()
    print(f"\n  Revenue formula verification  (qty × unit_price × (1−discount))")
    print(f"  Mismatches : {mismatch}  {'✓ All rows consistent.' if mismatch == 0 else '⚠ Inconsistencies found.'}")

    print("\n  New derived columns added:")
    print("    • year, month, year_month  (from order_date)")
    print("    • discount_band  : Low (0–10%) / Medium (11–20%) / High (21–35%)")
    print("    • delivery_band  : Fast (1–3 d) / Normal (4–7 d) / Slow (8–11 d)")
    print("    • rating_band    : Poor (1–2) / Neutral (3) / Good (4–5)")

    subsection("Discount Band Distribution")
    print(df["discount_band"].value_counts().sort_index().to_string())

    subsection("Delivery Band Distribution")
    print(df["delivery_band"].value_counts().sort_index().to_string())

    subsection("Rating Band Distribution")
    print(df["rating_band"].value_counts().sort_index().to_string())

    # ── Section 3 │ KPI Calculations ─────────────────────────────────────────
    section("SECTION 3 │ KPI CALCULATIONS")

    kpis = compute_kpis(df)

    subsection("LEVEL 1 – What is Happening?")
    print(f"  Total Revenue          : ${kpis['total_revenue']:>14,.2f}")
    print(f"  Total Orders           : {kpis['total_orders']:>14,}")
    print(f"  Total Customers        : {kpis['total_customers']:>14,}")
    print(f"  Average Order Value    : ${kpis['aov']:>14,.2f}")
    print(f"  Avg Customer Rating    : {kpis['avg_rating']:>14.2f} / 5.0")
    print(f"  Avg Delivery Days      : {kpis['avg_delivery_days']:>14.2f} days")

    subsection("LEVEL 3 – Revenue by Category")
    for cat, val in kpis["rev_by_category"].items():
        pct = val / kpis["total_revenue"] * 100
        print(f"  {cat:<15}  ${val:>12,.2f}  ({pct:.1f}%)")

    subsection("LEVEL 3 – Revenue by Region")
    for reg, val in kpis["rev_by_region"].items():
        pct = val / kpis["total_revenue"] * 100
        print(f"  {reg:<10}  ${val:>12,.2f}  ({pct:.1f}%)")

    subsection("LEVEL 3 – AOV by Category")
    for cat, val in kpis["aov_by_category"].items():
        print(f"  {cat:<15}  ${val:>8,.2f}")

    subsection("LEVEL 3 – Avg Rating by Category")
    for cat, val in kpis["avg_rating_by_cat"].items():
        print(f"  {cat:<15}  {val:.3f}")

    subsection("LEVEL 3 – Avg Rating by Region")
    for reg, val in kpis["avg_rating_by_reg"].items():
        print(f"  {reg:<10}  {val:.3f}")

    subsection("LEVEL 3 – Avg Delivery Days by Region")
    for reg, val in kpis["avg_del_by_region"].items():
        print(f"  {reg:<10}  {val:.2f} days")

    subsection("LEVEL 3 – Avg Revenue by Discount Band")
    for band, val in kpis["rev_by_disc_band"].items():
        print(f"  {str(band):<22}  ${val:>8,.2f}")

    subsection("LEVEL 3 – Avg Rating by Discount Band")
    for band, val in kpis["rat_by_disc_band"].items():
        print(f"  {str(band):<22}  {val:.3f}")

    subsection("LEVEL 4 – Risks & Opportunities")
    print(f"  Repeat Customer Rate        : {kpis['repeat_rate']:.1f}%  ({kpis['repeat_customers']:,} of {kpis['total_customers']:,} customers)")
    print(f"  Top 10% Customer Rev Share  : {kpis['top10pct_share']:.1f}%  (top {kpis['top10pct_n']} customers)")
    print(f"  Poor Rating Rate (≤ 2)      : {kpis['poor_rating_rate']:.1f}%  ({kpis['poor_rating_n']:,} orders)")
    print(f"  Slow Delivery Rate (≥ 9 d)  : {kpis['slow_delivery_rate']:.1f}%  ({kpis['slow_delivery_n']:,} orders)")
    print(f"  High Discount Rate (≥ 25%)  : {kpis['high_disc_rate']:.1f}%  ({kpis['high_disc_n']:,} orders)")

    # ── Section 4 │ Business Questions ───────────────────────────────────────
    section("SECTION 4 │ BUSINESS QUESTIONS ANALYSIS (Q1–Q10)")

    annual_revenue  = kpis["annual_revenue"]
    aov_by_category = kpis["aov_by_category"]
    rev_by_disc_band = kpis["rev_by_disc_band"]
    rat_by_disc_band = kpis["rat_by_disc_band"]
    aov             = kpis["aov"]
    total_revenue   = kpis["total_revenue"]
    total_customers = kpis["total_customers"]
    repeat_customers = kpis["repeat_customers"]
    repeat_rate     = kpis["repeat_rate"]
    cust_revenue    = kpis["cust_revenue"]
    avg_rating_by_cat = kpis["avg_rating_by_cat"]
    avg_rating_by_reg = kpis["avg_rating_by_reg"]
    aov_by_pm       = kpis["aov_by_pm"]

    questions = [
        ("Q1", "How is total revenue distributed across product categories?",
         kpis["rev_by_category"].to_dict()),
        ("Q2", "Which regions generate the most revenue and order volume?",
         {r: {"revenue": f"${v:,.2f}", "orders": int(df[df.region == r].shape[0])}
          for r, v in kpis["rev_by_region"].items()}),
        ("Q3", "What is the overall AOV, and how does it vary by category and region?",
         {"overall_AOV": f"${aov:,.2f}",
          "by_category": {k: f"${v:,.2f}" for k, v in aov_by_category.items()}}),
        ("Q4", "How does discount level relate to revenue and customer rating?",
         {"avg_revenue_by_disc": {str(k): f"${v:,.2f}" for k, v in rev_by_disc_band.items()},
          "avg_rating_by_disc":  {str(k): f"{v:.3f}" for k, v in rat_by_disc_band.items()}}),
        ("Q5", "Does delivery time affect customer satisfaction?",
         df.groupby("delivery_band", observed=True)["customer_rating"].mean()
           .round(3).to_dict()),
        ("Q6", "Which payment methods are most used, and do they correlate with higher AOV?",
         {pm: {"orders": int(df[df.payment_method == pm].shape[0]),
               "avg_revenue": f"${df[df.payment_method == pm]['revenue'].mean():,.2f}"}
          for pm in df["payment_method"].unique()}),
        ("Q7", "What percentage of customers are repeat buyers?",
         {"total_customers": total_customers,
          "repeat_customers": int(repeat_customers),
          "repeat_rate_pct": f"{repeat_rate:.1f}%",
          "top_customer_by_revenue": f"Customer {cust_revenue.index[0]} — ${cust_revenue.iloc[0]:,.2f}"}),
        ("Q8", "Which categories and regions have the weakest customer ratings?",
         {"worst_category": avg_rating_by_cat.idxmin(),
          "worst_region":   avg_rating_by_reg.idxmin(),
          "ratings_by_category": avg_rating_by_cat.round(3).to_dict(),
          "ratings_by_region":   avg_rating_by_reg.round(3).to_dict()}),
        ("Q9", "How does revenue trend over time (annual)?",
         {str(yr): f"${val:,.2f}" for yr, val in annual_revenue.items()}),
        ("Q10", "What specific business actions are recommended?",
         "→ See Section 6: Business Insights Framework"),
    ]

    for q_id, question, answer in questions:
        print(f"\n  [{q_id}] {question}")
        if isinstance(answer, dict):
            for k, v in answer.items():
                print(f"       {k}: {v}")
        else:
            print(f"       {answer}")

    # ── Section 5 │ Visualisations ────────────────────────────────────────────
    section("SECTION 5 │ VISUALISATIONS")
    generate_charts(df, kpis)

    # ── Section 6 │ Business Insights Framework ───────────────────────────────
    section("SECTION 6 │ BUSINESS INSIGHTS FRAMEWORK")
    print("  (Observation → Insight → Business Implication → Recommended Action)\n")

    for ins in build_insights(kpis):
        print(f"  +- [{ins['id']}] {ins['topic']}")
        print(f"  |  OBSERVATION  : {ins['observation']}")
        print(f"  |  INSIGHT      : {ins['insight']}")
        print(f"  |  IMPLICATION  : {ins['implication']}")
        print(f"  +- ACTION       : {ins['action']}")
        print()

    # ── Section 7 │ Dataset Limitations ──────────────────────────────────────
    section("SECTION 7 │ DATASET LIMITATIONS")

    limitations = [
        ("Synthetic / simulated data",
         "The dataset is entirely computer-generated. All patterns, values, and distributions are artificial."),
        ("Future-dated orders (3,273 of 5,000 orders)",
         "65% of orders have order_date beyond the current date, extending to September 2035."),
        ("Exactly one order per calendar day",
         "Every calendar day from 2022-01-01 to 2035-09-09 contains precisely one order."),
        ("Artificially uniform distributions",
         "Quantity (1–7), discount (0.00–0.35), and regional order counts are nearly equally distributed."),
        ("Flat revenue across categories",
         "All four product categories have nearly identical average revenue (~$1,000–$1,059)."),
        ("Closed customer pool (989 customers, IDs 1000–1999)",
         "No customer acquisition or churn can be observed."),
        ("No cost, margin, or external data",
         "Profitability, gross margin, return rates, campaign attribution, and competitor benchmarks cannot be computed."),
    ]

    for i, (title, detail) in enumerate(limitations, 1):
        print(f"\n  L{i}. {title}")
        print(f"     {detail}")

    print(f"\n{SEP}")
    print("  ANALYSIS COMPLETE")
    print(f"  All 15 charts saved to: ./{OUTPUT_DIR}/")
    print(f"  Pages: page1_executive_overview | page2_sales_product | page3_customer_operations")
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit dashboard
# Runs when: streamlit run ecommerce_analytics.py
# ─────────────────────────────────────────────────────────────────────────────

def run_streamlit():
    import streamlit as st

    # ── Page config ───────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="E-Commerce Analytics Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Cached data loader ────────────────────────────────────────────────────
    @st.cache_data(show_spinner="Loading and processing data…")
    def get_data():
        df   = load_and_prepare_data()
        kpis = compute_kpis(df)
        # Ensure charts exist on disk
        generate_charts(df, kpis)
        return df, kpis

    df, kpis = get_data()
    insights = build_insights(kpis)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
            width=80,
        )
        st.title("E-Commerce Analytics")
        st.caption("IBM SkillsBuild — Data Analytics with AI")
        st.divider()

        sections = [
            "📋 Executive Overview",
            "📦 Sales & Product Analysis",
            "👥 Customer & Operations",
            "💡 Business Insights",
            "✅ Recommendations",
        ]
        nav = st.radio("Navigate to", sections, label_visibility="collapsed")

        st.divider()
        st.markdown(
            f"**Dataset:** `{DATA_FILE}`  \n"
            f"**Records:** {kpis['total_orders']:,}  \n"
            f"**Date range:** 2022 – 2035  \n"
        )
        st.caption("⚠️ Synthetic dataset — illustrative purposes only.")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def metric_card(col, label, value, delta=None, delta_color="normal"):
        col.metric(label=label, value=value, delta=delta, delta_color=delta_color)

    def img(path, caption=None):
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.warning(f"Chart not found: {path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — EXECUTIVE OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════════
    if nav == "📋 Executive Overview":
        st.title("📋 Executive Overview")
        st.markdown(
            "> **Business Intelligence summary:** Revenue, order, and customer performance "
            "at a glance across the full dataset period (2022–2035)."
        )
        st.divider()

        # ── KPI strip ────────────────────────────────────────────────────────
        st.subheader("Key Performance Indicators")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        metric_card(c1, "Total Revenue",       f"${kpis['total_revenue']:,.0f}")
        metric_card(c2, "Total Orders",        f"{kpis['total_orders']:,}")
        metric_card(c3, "Total Customers",     f"{kpis['total_customers']:,}")
        metric_card(c4, "Avg Order Value",     f"${kpis['aov']:,.2f}")
        metric_card(c5, "Avg Customer Rating", f"{kpis['avg_rating']:.2f} / 5.0")
        metric_card(c6, "Avg Delivery Days",   f"{kpis['avg_delivery_days']:.1f} days")

        st.divider()

        # ── Secondary KPIs ────────────────────────────────────────────────────
        st.subheader("Operational KPIs")
        k1, k2, k3, k4, k5 = st.columns(5)
        metric_card(k1, "Repeat Customer Rate",    f"{kpis['repeat_rate']:.1f}%")
        metric_card(k2, "Top 10% Rev Share",       f"{kpis['top10pct_share']:.1f}%")
        metric_card(k3, "Poor Rating Rate (≤2)",   f"{kpis['poor_rating_rate']:.1f}%")
        metric_card(k4, "Slow Delivery Rate (≥9d)",f"{kpis['slow_delivery_rate']:.1f}%")
        metric_card(k5, "High Discount Rate (≥25%)",f"{kpis['high_disc_rate']:.1f}%")

        st.divider()

        # ── Charts ────────────────────────────────────────────────────────────
        st.subheader("Revenue Trends & Distribution")

        st.markdown("**Monthly Revenue Trend (2022–2035)**")
        img(os.path.join(PAGE1_DIR, "01_monthly_revenue_trend.png"))

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Revenue by Product Category**")
            img(os.path.join(PAGE1_DIR, "02_revenue_by_category.png"))
        with col_b:
            st.markdown("**Revenue by Region**")
            img(os.path.join(PAGE1_DIR, "03_revenue_by_region.png"))

        st.markdown("**Orders by Payment Method**")
        col_pie, col_space = st.columns([1, 1])
        with col_pie:
            img(os.path.join(PAGE1_DIR, "04_orders_by_payment_method.png"))

        st.divider()

        # ── Annual revenue table ───────────────────────────────────────────────
        st.subheader("Annual Revenue Summary")
        annual_df = kpis["annual_revenue"].reset_index()
        annual_df.columns = ["Year", "Revenue ($)"]
        annual_df["Revenue ($)"] = annual_df["Revenue ($)"].map("${:,.2f}".format)
        st.dataframe(annual_df, use_container_width=False, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — SALES & PRODUCT ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    elif nav == "📦 Sales & Product Analysis":
        st.title("📦 Sales & Product Analysis")
        st.markdown(
            "> Analysis of discount effectiveness, average order values, category revenue "
            "share, and unit price distributions across product categories."
        )
        st.divider()

        # ── KPIs ──────────────────────────────────────────────────────────────
        st.subheader("Category & Discount KPIs")
        top_cat = kpis["rev_by_category"].idxmax()
        top_cat_rev = kpis["rev_by_category"].max()
        top_cat_pct = top_cat_rev / kpis["total_revenue"] * 100
        top_aov_cat = kpis["aov_by_category"].idxmax()

        d1, d2, d3, d4 = st.columns(4)
        metric_card(d1, "Top Revenue Category",   top_cat)
        metric_card(d2, "Top Category Revenue",   f"${top_cat_rev:,.0f}")
        metric_card(d3, "Top Category Share",     f"{top_cat_pct:.1f}%")
        metric_card(d4, "Highest AOV Category",   top_aov_cat)

        st.divider()

        # ── Discount analysis ─────────────────────────────────────────────────
        st.subheader("Discount Band Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Avg Revenue by Discount Band**")
            img(os.path.join(PAGE2_DIR, "05_avg_revenue_by_discount_band.png"))
        with col2:
            st.markdown("**Avg Rating by Discount Band**")
            img(os.path.join(PAGE2_DIR, "06_avg_rating_by_discount_band.png"))

        disc_table = pd.DataFrame({
            "Discount Band": [str(b) for b in kpis["rev_by_disc_band"].index],
            "Avg Revenue ($)": [f"${v:,.2f}" for v in kpis["rev_by_disc_band"].values],
            "Avg Rating":      [f"{v:.3f}" for v in kpis["rat_by_disc_band"].values],
        })
        st.dataframe(disc_table, use_container_width=False, hide_index=True)

        st.divider()

        # ── AOV heatmap ───────────────────────────────────────────────────────
        st.subheader("AOV by Category × Region")
        img(os.path.join(PAGE2_DIR, "07_aov_category_x_region_heatmap.png"))

        st.divider()

        # ── Revenue share & unit price ────────────────────────────────────────
        st.subheader("Revenue Share & Pricing")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Annual Revenue by Category (Stacked)**")
            img(os.path.join(PAGE2_DIR, "08_category_revenue_stacked_area.png"))
        with col4:
            st.markdown("**Unit Price Distribution by Category**")
            img(os.path.join(PAGE2_DIR, "09_unit_price_distribution_by_category.png"))

        st.divider()

        # ── AOV by category table ──────────────────────────────────────────────
        st.subheader("Average Order Value by Category")
        aov_df = kpis["aov_by_category"].reset_index()
        aov_df.columns = ["Category", "Avg Order Value ($)"]
        aov_df["Avg Order Value ($)"] = aov_df["Avg Order Value ($)"].map("${:,.2f}".format)
        st.dataframe(aov_df, use_container_width=False, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — CUSTOMER & OPERATIONS ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    elif nav == "👥 Customer & Operations":
        st.title("👥 Customer & Operations Analysis")
        st.markdown(
            "> Customer loyalty, top-revenue customers, delivery performance, and "
            "satisfaction metrics by region, category, and payment method."
        )
        st.divider()

        # ── Customer KPIs ─────────────────────────────────────────────────────
        st.subheader("Customer KPIs")
        top_cust_id  = kpis["cust_revenue"].index[0]
        top_cust_rev = kpis["cust_revenue"].iloc[0]
        worst_cat_rat = kpis["avg_rating_by_cat"].idxmin()
        worst_reg_rat = kpis["avg_rating_by_reg"].idxmin()

        ck1, ck2, ck3, ck4 = st.columns(4)
        metric_card(ck1, "Repeat Customer Rate",  f"{kpis['repeat_rate']:.1f}%")
        metric_card(ck2, "Top Customer Revenue",  f"${top_cust_rev:,.2f}")
        metric_card(ck3, "Lowest Rated Category", worst_cat_rat)
        metric_card(ck4, "Lowest Rated Region",   worst_reg_rat)

        st.divider()

        # ── Customer charts ───────────────────────────────────────────────────
        st.subheader("Customer Behaviour")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Customer Order Frequency Distribution**")
            img(os.path.join(PAGE3_DIR, "10_customer_order_frequency.png"))
        with col2:
            st.markdown("**Top 20 Customers by Total Revenue**")
            img(os.path.join(PAGE3_DIR, "11_top20_customers_by_revenue.png"))

        st.divider()

        # ── Delivery & satisfaction ────────────────────────────────────────────
        st.subheader("Delivery & Satisfaction")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Avg Rating by Delivery Speed**")
            img(os.path.join(PAGE3_DIR, "12_avg_rating_by_delivery_band.png"))
        with col4:
            st.markdown("**Avg Delivery Days by Region**")
            img(os.path.join(PAGE3_DIR, "13_avg_delivery_days_by_region.png"))

        st.divider()

        # ── Satisfaction heatmap & payment AOV ────────────────────────────────
        st.subheader("Satisfaction & Payment Analysis")
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("**Avg Rating — Category × Region**")
            img(os.path.join(PAGE3_DIR, "14_avg_rating_category_x_region_heatmap.png"))
        with col6:
            st.markdown("**AOV by Payment Method**")
            img(os.path.join(PAGE3_DIR, "15_aov_by_payment_method.png"))

        st.divider()

        # ── Regional & payment tables ─────────────────────────────────────────
        st.subheader("Regional & Payment Summary Tables")
        t1, t2 = st.columns(2)

        with t1:
            st.markdown("**Delivery Days by Region**")
            del_df = kpis["avg_del_by_region"].reset_index()
            del_df.columns = ["Region", "Avg Delivery Days"]
            del_df["Avg Delivery Days"] = del_df["Avg Delivery Days"].map("{:.2f}".format)
            st.dataframe(del_df, use_container_width=False, hide_index=True)

        with t2:
            st.markdown("**AOV & Rating by Payment Method**")
            pm_df = pd.DataFrame({
                "Payment Method": kpis["aov_by_pm"].index,
                "Avg Order Value": [f"${v:,.2f}" for v in kpis["aov_by_pm"].values],
                "Order Count":     [str(kpis["pm_counts"].get(pm, 0)) for pm in kpis["aov_by_pm"].index],
            })
            st.dataframe(pm_df, use_container_width=False, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 4 — BUSINESS INSIGHTS
    # ═══════════════════════════════════════════════════════════════════════════
    elif nav == "💡 Business Insights":
        st.title("💡 Business Insights")
        st.markdown(
            "> Each insight follows the framework:  \n"
            "> **Observation → Insight → Business Implication → Recommended Action**"
        )
        st.divider()

        for ins in insights:
            with st.expander(f"**[{ins['id']}] {ins['topic']}**", expanded=True):
                col_l, col_r = st.columns([1, 3])
                with col_l:
                    st.markdown(f"### `{ins['id']}`")
                    st.markdown(f"**{ins['topic']}**")
                with col_r:
                    st.markdown(f"🔎 **Observation**")
                    st.info(ins["observation"])
                    st.markdown(f"💡 **Insight**")
                    st.write(ins["insight"])
                    st.markdown(f"⚠️ **Business Implication**")
                    st.warning(ins["implication"])
                    st.markdown(f"✅ **Recommended Action**")
                    st.success(ins["action"])

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 5 — RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    elif nav == "✅ Recommendations":
        st.title("✅ Recommendations")
        st.markdown(
            "> Prioritised, actionable recommendations derived from the full analysis. "
            "Each recommendation is tied directly to a data finding."
        )
        st.divider()

        recommendations = [
            {
                "priority": "🔴 High",
                "title":    "Improve Delivery Speed",
                "finding":  f"Slow deliveries (8–11 days) generate significantly lower customer ratings ({kpis['rat_by_del'].get('Slow (8–11 days)', 0):.2f}/5.0) compared to fast deliveries ({kpis['rat_by_del'].get('Fast (1–3 days)', 0):.2f}/5.0).",
                "action":   f"Set a maximum 7-day delivery SLA. Invest in logistics for the slowest region: {kpis['avg_del_by_region'].idxmax()} ({kpis['avg_del_by_region'].max():.1f} days avg).",
                "metric":   f"Target: avg delivery ≤ 5 days; customer rating ≥ 4.0",
            },
            {
                "priority": "🔴 High",
                "title":    "Protect Top 10% Customer Revenue",
                "finding":  f"The top {kpis['top10pct_n']} customers account for {kpis['top10pct_share']:.1f}% of total revenue. Losing even a few of these customers would have a disproportionate impact.",
                "action":   "Launch a tiered loyalty programme with personalised offers, priority support, and early product access for top-revenue customers.",
                "metric":   "Target: top customer retention rate ≥ 95%",
            },
            {
                "priority": "🟡 Medium",
                "title":    "Revise Discount Strategy",
                "finding":  f"High discounts (21–35%) do not meaningfully improve customer ratings ({kpis['rat_by_disc_band'].get('High (21–35%)', 0):.2f}/5.0 vs {kpis['rat_by_disc_band'].get('Low (0–10%)', 0):.2f}/5.0 for low discounts) but erode margins.",
                "action":   "Cap maximum discount at 20%. Reallocate promotional budget to delivery speed improvements and product quality — the primary satisfaction drivers.",
                "metric":   "Target: reduce high-discount orders below 10% of total",
            },
            {
                "priority": "🟡 Medium",
                "title":    "Address Satisfaction Weak Spots",
                "finding":  f"Category '{kpis['avg_rating_by_cat'].idxmin()}' and region '{kpis['avg_rating_by_reg'].idxmin()}' show the lowest customer ratings. {kpis['poor_rating_rate']:.1f}% of orders have a rating ≤ 2.",
                "action":   f"Conduct root-cause analysis for {kpis['avg_rating_by_cat'].idxmin()} products in {kpis['avg_rating_by_reg'].idxmin()}. Investigate product quality, delivery, or pricing as the driver.",
                "metric":   f"Target: reduce poor rating rate (≤2) below 5%",
            },
            {
                "priority": "🟡 Medium",
                "title":    "Diversify Revenue Across Categories",
                "finding":  f"{kpis['rev_by_category'].idxmax()} contributes {kpis['rev_by_category'].max() / kpis['total_revenue'] * 100:.1f}% of total revenue. Revenue concentration increases business risk.",
                "action":   "Increase marketing investment and inventory for under-performing categories. Set a category revenue balance target (no single category > 30% of revenue).",
                "metric":   "Target: no single category > 30% of total revenue",
            },
            {
                "priority": "🟢 Low",
                "title":    "Leverage Payment Method Segmentation",
                "finding":  f"Card users have the highest AOV (${kpis['aov_by_pm'].get('Card', 0):,.0f}) while Wallet users have the lowest (${kpis['aov_by_pm'].get('Wallet', 0):,.0f}).",
                "action":   "Offer premium, high-value product bundles to card users. Introduce wallet cashback incentives to increase basket size for wallet users.",
                "metric":   "Target: lift wallet user AOV by 15% within 2 quarters",
            },
            {
                "priority": "🟢 Low",
                "title":    "Regional Expansion Strategy",
                "finding":  f"Revenue is relatively balanced across regions ({kpis['rev_by_region'].idxmax()} leads at {kpis['rev_by_region'].max() / kpis['total_revenue'] * 100:.1f}%). No clear high-growth region is visible from revenue data alone.",
                "action":   "Cross-reference revenue with customer satisfaction and delivery performance per region to identify the best expansion targets. Pilot targeted campaigns in top-satisfaction regions.",
                "metric":   "Target: identify and invest in top 2 expansion regions within Q2",
            },
        ]

        for rec in recommendations:
            with st.expander(f"{rec['priority']} — **{rec['title']}**", expanded=True):
                rc1, rc2 = st.columns([1, 3])
                with rc1:
                    st.markdown(f"**Priority**  \n{rec['priority']}")
                with rc2:
                    st.markdown("📊 **Data Finding**")
                    st.info(rec["finding"])
                    st.markdown("✅ **Recommended Action**")
                    st.success(rec["action"])
                    st.markdown("📏 **Success Metric**")
                    st.write(f"`{rec['metric']}`")

        st.divider()
        st.subheader("⚠️ Dataset Limitations")
        st.markdown(
            """
This dashboard is built on a **synthetic / simulated dataset**. Please note the following
limitations before using these findings to drive real business decisions:

| # | Limitation | Detail |
|---|-----------|--------|
| L1 | **Synthetic data** | All patterns, values, and distributions are computer-generated. |
| L2 | **Future-dated orders** | 65% of orders (3,273/5,000) are dated beyond today, up to Sept 2035. |
| L3 | **One order per day** | Every calendar day has exactly one order — no seasonality or demand spikes. |
| L4 | **Uniform distributions** | Quantity, discount, and region counts are artificially balanced. |
| L5 | **Flat category revenue** | All categories have near-identical average revenue (~$1,000–$1,059). |
| L6 | **Closed customer pool** | 989 customers (IDs 1000–1999) — no acquisition or churn modelling possible. |
| L7 | **No cost/margin data** | Profitability, ROI, and campaign attribution cannot be computed. |
"""
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "E-Commerce Sales & Customer Analytics Dashboard · "
        "IBM SkillsBuild Data Analytics with AI · "
        "Synthetic dataset — for illustrative purposes only."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — detect Streamlit vs plain Python execution
# ─────────────────────────────────────────────────────────────────────────────

def _is_streamlit() -> bool:
    """Return True when this module is being served by Streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _is_streamlit():
    run_streamlit()
else:
    run_cli()
