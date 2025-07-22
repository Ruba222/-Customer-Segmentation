import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import datetime as dt # Import datetime for date calculations

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def app():
    st.set_page_config(layout="wide", page_title="Customer Segmentation App (DBSCAN)")

    st.title("📊 Customer Segmentation Visualizations")
    st.markdown("Explore the distributions of Recency, Orders, and Price, and visualize the customer segments in 3D space.")

    # Retrieve data from session state
    if 'rfm_df' not in st.session_state or 'rfm_scaled_df' not in st.session_state or 'user_label_map' not in st.session_state or 'processed_df' not in st.session_state or 'snapshot_date' not in st.session_state:
        st.warning("Please go to the 'Prediction' page first to load the data and train the model.")
        return

    rfm_df = st.session_state['rfm_df']
    rfm_scaled_df = st.session_state['rfm_scaled_df']
    user_label_map = st.session_state['user_label_map']
    processed_df = st.session_state['processed_df']
    snapshot_date = st.session_state['snapshot_date'] # Retrieve snapshot_date

    # Ensure Segment_Label is mapped for visualization
    rfm_df['Segment_Label'] = rfm_df['Cluster'].map(user_label_map)
    rfm_scaled_df['Segment_Label'] = rfm_scaled_df['Cluster'].map(user_label_map)

    # --- Customer Activity by Last Purchase Month Plot ---
    st.header("Customer Activity by Last Purchase Month")
    # st.write("This plot shows the number of customers whose *last purchase* occurred in each month.")
    st.markdown("""
       ### 📊 Interpretation of the Plot

       In **November 2010**, there was a **notable spike** in customer activity, with over **1,400 customers** making purchases.

       #### Possible Reasons:
        - 🎄 **Holiday Season**: November leads into Christmas, encouraging early gift shopping.
        - 💸 **Sales Events**: Promotions like **Black Friday** likely boosted purchases.
        - 📢 **Campaigns**: Effective marketing or discounts could explain the surge.

👉 This trend shows the **impact of seasonality** on customer behavior.
""")


    # Calculate lastPurchaseDate for each customer based on Recency and snapshot_date
    customer_last_purchase_data = rfm_df[['Customer ID', 'Recency']].copy()
    customer_last_purchase_data['lastPurchaseDate'] = snapshot_date - pd.to_timedelta(customer_last_purchase_data['Recency'], unit='D')

    # Extract Month from lastPurchaseDate
    customer_last_purchase_data['Month'] = customer_last_purchase_data['lastPurchaseDate'].dt.to_period('M')

    # Group by Month and count customers
    monthly_counts = customer_last_purchase_data.groupby('Month')['Customer ID'].count().reset_index()
    monthly_counts.columns = ['Month', 'CustomerCount']
    monthly_counts['Month'] = monthly_counts['Month'].astype(str) # Convert Period to string for plotting

    fig_last_purchase_month, ax_last_purchase_month = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=monthly_counts, x='Month', y='CustomerCount', marker='o', ax=ax_last_purchase_month)
    ax_last_purchase_month.set_title('Customer Activity by Last Purchase Month')
    ax_last_purchase_month.set_xlabel('Month')
    ax_last_purchase_month.set_ylabel('Customer Count (Last Purchase)')
    ax_last_purchase_month.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_last_purchase_month)

    st.markdown("---")

    # --- Distribution Plots ---
    st.header("Distribution of Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Recency Distribution")
        fig_recency, ax_recency = plt.subplots(figsize=(7, 5))
        sns.histplot(rfm_df['Recency'], kde=True, ax=ax_recency, color='skyblue')
        ax_recency.set_title('Distribution of Recency')
        ax_recency.set_xlabel('Recency (Days)')
        ax_recency.set_ylabel('Number of Customers')
        st.pyplot(fig_recency)

    with col2:
        st.subheader("Orders Distribution")
        fig_orders, ax_orders = plt.subplots(figsize=(7, 5))
        sns.histplot(rfm_df['Orders'], kde=True, ax=ax_orders, color='lightcoral')
        ax_orders.set_title('Distribution of Orders')
        ax_orders.set_xlabel('Orders (Number of Invoices)')
        ax_orders.set_ylabel('Number of Customers')
        st.pyplot(fig_orders)

    with col3:
        st.subheader("Price Distribution")
        fig_price, ax_price = plt.subplots(figsize=(7, 5))
        sns.histplot(rfm_df['Price'], kde=True, ax=ax_price, color='lightgreen')
        ax_price.set_title('Distribution of Price')
        ax_price.set_xlabel('Price (Total Spent)')
        ax_price.set_ylabel('Number of Customers')
        st.pyplot(fig_price)

    st.markdown("---")

    # --- Box Plots by Cluster ---
    st.header("Feature Box Plots by Segment")

    col_b1, col_b2, col_b3 = st.columns(3)

    with col_b1:
        st.subheader("Recency by Segment")
        fig_box_recency, ax_box_recency = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=rfm_df, x='Segment_Label', y='Recency', ax=ax_box_recency, palette='viridis')
        ax_box_recency.set_title('Recency by Customer Segment')
        ax_box_recency.set_xlabel('Customer Segment')
        ax_box_recency.set_ylabel('Recency (Days)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_box_recency)

    with col_b2:
        st.subheader("Orders by Segment")
        fig_box_orders, ax_box_orders = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=rfm_df, x='Segment_Label', y='Orders', ax=ax_box_orders, palette='viridis')
        ax_box_orders.set_title('Orders by Customer Segment')
        ax_box_orders.set_xlabel('Customer Segment')
        ax_box_orders.set_ylabel('Orders (Number of Invoices)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_box_orders)

    with col_b3:
        st.subheader("Price by Segment")
        fig_box_price, ax_box_price = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=rfm_df, x='Segment_Label', y='Price', ax=ax_box_price, palette='viridis')
        ax_box_price.set_title('Price by Customer Segment')
        ax_box_price.set_xlabel('Customer Segment')
        ax_box_price.set_ylabel('Price (Total Spent)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_box_price)

    st.markdown("---")

    # --- 3D Scatter Plot ---
    st.header("3D Scatter Plot of Scaled Features by Segment")
    st.write("This plot visualizes customer segments in a 3D space based on their scaled Recency, Orders, and Price values.")

    fig_3d = plt.figure(figsize=(12, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    unique_labels = rfm_scaled_df['Segment_Label'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))

    for label in unique_labels:
        subset = rfm_scaled_df[rfm_scaled_df['Segment_Label'] == label]
        ax_3d.scatter(subset['Recency_Scaled'], subset['Orders_Scaled'], subset['Price_Scaled'],
                      label=label, color=label_to_color[label], s=70, alpha=0.7)

    ax_3d.set_title('Customer Segments (3D Scaled RFM Features)', fontsize=16)
    ax_3d.set_xlabel('Recency (Scaled)', fontsize=12)
    ax_3d.set_ylabel('Orders (Scaled)', fontsize=12)
    ax_3d.set_zlabel('Price (Scaled)', fontsize=12)
    ax_3d.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_3d)

if __name__ == "__main__":
    app()
