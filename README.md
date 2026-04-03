# E-Commerce Customer Segmentation with RFM Analysis

This project simulates e-commerce transactions, calculates RFM metrics, applies K-Means clustering, and maps each cluster to business-friendly customer segments for activation and retention campaigns.

## Features

- Generates 1,000+ synthetic transactions with `CustomerID`, `OrderDate`, and `TransactionAmount`
- Builds `Recency`, `Frequency`, and `Monetary` metrics per customer
- Adds quartile-based `R_Score`, `F_Score`, `M_Score`, and `RFM_Score`
- Normalizes features with `StandardScaler`
- Uses the elbow method to choose the K-Means cluster count automatically
- Assigns customer-friendly segments: `Champions`, `Loyal Customers`, `At-Risk`, and `Lost Customers`
- Prints a marketing strategy report
- Exports raw transactions, a BI-ready customer file, and a segment summary
- Saves elbow and cluster visualizations as PNG files
- Supports CLI arguments for customer volume, date range, cluster bounds, and output path

## Project Files

- [rfm_customer_segmentation.py](c:\Users\leowillfride\OneDrive\Desktop\RFM\rfm_customer_segmentation.py)
- [requirements.txt](c:\Users\leowillfride\OneDrive\Desktop\RFM\requirements.txt)

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python rfm_customer_segmentation.py
```

You can also customize the run:

```bash
python rfm_customer_segmentation.py --num-customers 500 --min-k 4 --max-k 7 --output-dir output
```

## Outputs

The script creates an `output/` folder with:

- `synthetic_transactions.csv`
- `customer_segments_bi_ready.csv`
- `segment_summary.csv`
- `customer_segments.png`
- `elbow_curve.png`

## Segment Logic

The project uses two layers of segmentation logic:

1. K-Means finds natural behavioral clusters in scaled RFM space.
2. Each cluster is mapped to a business segment using averaged quartile-based RFM scores.

The business mapping is intentionally opinionated:

- The strongest cluster by weighted RFM quality is labeled `Champions`
- The weakest cluster by weighted RFM quality is labeled `Lost Customers`
- Remaining clusters with weak recency but still meaningful value are labeled `At-Risk`
- All other mid-to-strong clusters are labeled `Loyal Customers`

Interpretation:

- `Champions`: very recent, frequent, and high-spending customers
- `Loyal Customers`: consistent repeat buyers with good engagement and spend
- `At-Risk`: customers with decent historical value whose purchase recency has declined
- `Lost Customers`: the least engaged and lowest-quality customer group

## BI Notes

The exported customer dataset includes raw RFM measures, quartile-based RFM scores, cluster id, business segment label, and cluster quality score.

The segment summary file includes customer count plus average recency, frequency, monetary value, and RFM score per segment.

This makes it easy to build dashboards, segment filters, retention views, and campaign tracking in Tableau or Power BI.
