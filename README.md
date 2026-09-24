# E-Commerce Sales & Customer Analytics

A data analytics project that analyses e-commerce sales and customer data to identify meaningful business insights and support data-driven decision-making.

---

## Project Overview

This project follows a Business Intelligence approach:

**Data → Information → Insights → Business Decisions → Actions**

The project analyses e-commerce transactions to understand sales performance, customer behaviour, product categories, regional performance, discounts, payment methods, customer ratings, and delivery performance.

The analysis is delivered through two interfaces:
- **CLI mode** – run `python ecommerce_analytics.py` to execute the full analysis and save all 15 charts to the `output/` folder
- **Streamlit dashboard** – run `streamlit run ecommerce_analytics.py` for an interactive business analytics dashboard

---

## Business Problem / Objective

The main objective is to transform raw e-commerce data into meaningful business insights that help organisations understand:

1. **KPIs** – What is happening in the business?
2. **Trends** – How is performance changing over time?
3. **Drivers** – Which factors are associated with higher or lower performance?
4. **Risks / Opportunities** – What should the business pay attention to?
5. **Actions** – What evidence-based decisions can be considered?

The goal is not simply to create charts, but to generate actionable business insights from the data.

---

## Dataset

### File

`ecommerce_sales_analytics_5000.csv`

### Description

A **synthetic (computer-generated)** e-commerce sales simulation containing **5,000 order records** spanning January 2022 to September 2035. The data is entirely artificial — no finding should be treated as a real-world business fact.

### Source

Kaggle: [E-Commerce Sales Dataset](https://www.kaggle.com/datasets/abbas829/ecommerce-sales-dataset)

### Columns

| Column | Type | Description |
|--------|------|-------------|
| order_id | Integer | Unique order identifier |
| order_date | Date | Order date (2022-01-01 to 2035-09-09) |
| customer_id | Integer | Customer identifier (989 unique customers) |
| product_category | String | Electronics, Clothing, Home, Beauty |
| region | String | West, North, South, East |
| quantity | Integer | Units ordered (1–7) |
| unit_price | Float | Price per unit |
| discount | Float | Discount rate (0.00–0.35) |
| payment_method | String | Card, COD, Wallet |
| delivery_days | Integer | Days from order to delivery (1–11) |
| customer_rating | Float | Post-delivery rating (1.0–5.0) |
| revenue | Float | quantity × unit_price × (1 − discount) |

---

## Key Areas of Analysis

### 1. Executive Overview
- Total Revenue: $5,109,775.74
- Total Orders: 5,000
- Total Customers: 989
- Average Order Value (AOV): $1,021.96
- Average Customer Rating: 2.97 / 5.0
- Average Delivery Days: 6.12

### 2. Sales & Product Analysis
- Revenue by product category
- Discount band analysis (revenue and rating by discount level)
- AOV by category and region (heatmap)
- Annual revenue by category (stacked area)
- Unit price distribution by category

### 3. Customer Analysis
- Customer order frequency distribution
- Top 20 customers by total revenue
- Repeat customer rate (96.1%)

### 4. Operations Analysis
- Average rating by delivery speed
- Average delivery days by region
- Rating by category × region (heatmap)
- AOV by payment method

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3 | Core analysis language |
| Pandas | Data loading, cleaning, transformation, and aggregation |
| Matplotlib | Chart generation |
| Seaborn | Statistical visualisation (heatmaps, box plots) |
| Streamlit 1.x | Interactive web dashboard |

---

## Project Structure

```text
Ecommerce-Sales-Customer-Analytics/
│
├── ecommerce_analytics.py          # Main script — CLI analysis + Streamlit dashboard
├── ecommerce_sales_analytics_5000.csv  # Dataset (synthetic)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── report/
│   ├── Project_Report_Final.docx  # Final project report with charts and insights
│   └── Project_Report.md          # Original markdown report draft
│
└── output/
    ├── page1_executive_overview/
    │   ├── 01_monthly_revenue_trend.png
    │   ├── 02_revenue_by_category.png
    │   ├── 03_revenue_by_region.png
    │   └── 04_orders_by_payment_method.png
    ├── page2_sales_product/
    │   ├── 05_avg_revenue_by_discount_band.png
    │   ├── 06_avg_rating_by_discount_band.png
    │   ├── 07_aov_category_x_region_heatmap.png
    │   ├── 08_category_revenue_stacked_area.png
    │   └── 09_unit_price_distribution_by_category.png
    └── page3_customer_operations/
        ├── 10_customer_order_frequency.png
        ├── 11_top20_customers_by_revenue.png
        ├── 12_avg_rating_by_delivery_band.png
        ├── 13_avg_delivery_days_by_region.png
        ├── 14_avg_rating_category_x_region_heatmap.png
        └── 15_aov_by_payment_method.png
```

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Ecommerce-Sales-Customer-Analytics.git
cd Ecommerce-Sales-Customer-Analytics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the CLI analysis

Generates all 15 charts to the `output/` folder and prints the full analysis to the terminal:

```bash
python ecommerce_analytics.py
```

### 4. Run the Streamlit dashboard

```bash
streamlit run ecommerce_analytics.py
```

Opens the interactive business analytics dashboard in your browser at `http://localhost:8501`.

---

## Dataset Limitations

- The dataset is entirely **synthetic** — all patterns and values are computer-generated
- Future-dated orders (65% of records) extend to September 2035
- Exactly one order per calendar day — no real seasonality or demand spikes
- Uniformly distributed quantities, discounts, and regional counts
- No cost, margin, or profitability data available

---

## Notes

- This project was completed as part of the **IBM SkillsBuild Data Analytics with AI** internship programme
- All findings are illustrative and based on synthetic data only
- Do not use these results to inform real business decisions
