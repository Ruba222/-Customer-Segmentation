import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def app():
    st.set_page_config(layout="wide", page_title="Customer Segmentation App (DBSCAN)")

    st.title("📊 Customer Segment Summaries")
    st.markdown("This page provides detailed average RFM (Recency, Orders, Price) characteristics for each identified customer segment.")

    # Retrieve data from session state
    if 'rfm_df' not in st.session_state or 'user_label_map' not in st.session_state:
        st.warning("Please go to the 'Prediction' page first to load the data and train the model.")
        return

    rfm_df = st.session_state['rfm_df']
    user_label_map = st.session_state['user_label_map']

    # Ensure Segment_Label is mapped for display
    rfm_df['Segment_Label'] = rfm_df['Cluster'].map(user_label_map)

    st.header("Average Features Values by Segment")
    st.write("Review the average Recency, Orders, and Price for each customer segment to understand their typical behavior.")

    # Get unique cluster numbers and sort them, ensuring -1 (VIP Customers) is handled correctly
    unique_cluster_nums = sorted(rfm_df['Cluster'].unique(), key=lambda x: (x == -1, x))

    for cluster_num in unique_cluster_nums:
        label = user_label_map.get(cluster_num, f"Cluster {cluster_num} (Unmapped)")
        st.subheader(f"Segment: {label}")
        summary_df = rfm_df[rfm_df['Cluster'] == cluster_num]
        if not summary_df.empty:
            summary = summary_df[['Recency', 'Orders', 'Price']].mean().to_frame().T # Convert to DataFrame for better display
            summary.columns = ['Average Recency (Days)', 'Average Orders (Invoices)', 'Average Price (Total Spent)']
            st.dataframe(summary.style.format({"Average Recency (Days)": "{:.2f}", "Average Orders (Invoices)": "{:.2f}", "Average Price (Total Spent)": "£{:.2f}"}))
            st.write(f"**Number of customers in this segment:** {len(summary_df)}")

            st.markdown(f"**Description of '{label}':**")
            if label == 'VIP Customers':
                st.write("These customers are identified as VIPs. According to DBSCAN, these might be noise points or a distinct cluster depending on parameters. They are likely your most valuable customers, purchasing very recently, making many orders, and spending a significant amount (high price).")
            elif label == 'Churned Low Spenders':
                st.write("These customers have not purchased recently, make few orders, and spend little (low price). They are likely churned and were not highly profitable. Focus on understanding why they left, but prioritize higher-value segments for re-engagement.")
            elif label == 'Engaged High Spenders':
                st.write("These customers are highly engaged and contribute significantly to revenue. They make many orders and and spend well (high price). Nurture them to become VIPs.")
            elif label == 'Lost One-Time Buyers':
                st.write("These customers made a purchase a long time ago and haven't returned, often having made only one order with low price. Re-engagement might be challenging but worth a try with specific campaigns.")
            elif label == 'New High-Value Buyers':
                st.write("These are recent customers who have already shown good spending potential. They are important for growth; encourage repeat orders.")
            else:
                st.write("No specific description available for this cluster. This might indicate an unusual customer profile or a need to refine cluster definitions.")
        else:
            st.info("No customers found in this segment with current DBSCAN parameters.")
        st.markdown("---")

if __name__ == "__main__":
    app()
