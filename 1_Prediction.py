

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
FILE_PATH = 'online_retail_II.xlsx'

# --- User-defined Cluster Labels ---
USER_LABEL_MAP = {
    -1: 'VIP Customers',
     0: 'Churned Low Spenders',
     1: 'Engaged High Spenders',
     2: 'Lost One-Time Buyers',
     3: 'New High-Value Buyers'
}

# --- Data Loading and Preprocessing Function ---
@st.cache_data
def load_and_preprocess_data(file_path):
    """
    Loads the retail data and performs all necessary preprocessing steps.
    """
    try:
        data = pd.read_excel(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. "
                 "Please ensure it's in the same directory as the Streamlit app.")
        return None

    df = data.copy()

    # 1. Drop rows with missing 'Description'
    df.dropna(axis=0, how='any', subset=['Description'], inplace=True)

    # 2. Remove cancelled transactions (Invoice starting with 'C')
    df = df[~df['Invoice'].astype(str).str.startswith('C')]

    # 3. Remove transactions with negative 'Quantity'
    df = df[df['Quantity'] > 0]

    # 4. Remove transactions with Invoice starting with 'A' and null Customer ID (bad debt adjustments)
    df = df[~((df['Invoice'].astype(str).str.startswith('A')) & (df['Customer ID'].isnull()))]

    # 5. Handle duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 6. Fill missing 'Customer ID' with 'Invoice' number and ensure numeric type
    df['Customer ID'] = df['Customer ID'].fillna(df['Invoice']).astype(str)
    df['Customer ID'] = pd.to_numeric(df['Customer ID'], errors='coerce')
    df.dropna(subset=['Customer ID'], inplace=True)
    df['Customer ID'] = df['Customer ID'].astype(int)

    # 7. Calculate 'TotalPrice' for each transaction
    df['TotalPrice'] = df['Quantity'] * df['Price']

    return df

# --- RFM Feature Engineering Function ---
@st.cache_data
def calculate_rfm(df):
    """
    Calculates Recency, Orders, and Price features for each customer.
    """
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

    rfm_df = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()

    rfm_df.rename(columns={'InvoiceDate': 'Recency',
                           'Invoice': 'Orders',
                           'TotalPrice': 'Price'}, inplace=True)

    rfm_df = rfm_df[rfm_df['Price'] > 0]
    rfm_df = rfm_df[rfm_df['Orders'] > 0]

    return rfm_df, snapshot_date

# --- Clustering Function (DBSCAN) ---
@st.cache_resource
def train_dbscan_model(rfm_df_for_scaling, eps, min_samples):
    """
    Scales RFM features using RobustScaler and trains the DBSCAN clustering model.
    Returns the scaler and the trained DBSCAN model.
    """
    scaler = RobustScaler()
    rfm_scaled = scaler.fit_transform(rfm_df_for_scaling)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency_Scaled', 'Orders_Scaled', 'Price_Scaled'])

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(rfm_scaled_df)

    return scaler, dbscan, rfm_scaled_df

# --- Main Application Logic for Prediction Page ---
def main():
    st.set_page_config(layout="wide", page_title="Customer Segmentation App (DBSCAN)")
    st.image("assets/header.jpg", use_container_width=True)


    st.title("🛍️ Customer Segmentation and Prediction (DBSCAN)")
    st.markdown("This application helps you understand customer segments based on Recency, Orders and  Price analysis using **DBSCAN** clustering.")
    st.markdown("DBSCAN identifies clusters based on density, and can also identify 'noise' points that don't belong to any cluster.")

    # --- DBSCAN Parameters in Sidebar ---
    st.sidebar.header("DBSCAN Parameters")
    eps = 5
    min_samples = st.sidebar.slider("Min Samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.",
                                    min_value=1, max_value=50, value=5, step=1)

    # --- Data Loading and Model Training ---
    st.sidebar.header("Application Status")
    with st.spinner("Loading, preprocessing data, and training model... This may take a moment."):
        processed_df = load_and_preprocess_data(FILE_PATH)

        if processed_df is None:
            st.stop()

        rfm_df, snapshot_date = calculate_rfm(processed_df)
        rfm_features = rfm_df[['Recency', 'Orders', 'Price']]
        scaler, dbscan_model, rfm_scaled_df = train_dbscan_model(rfm_features, eps, min_samples)

        # Add cluster labels to the RFM DataFrame
        rfm_df['Cluster'] = dbscan_model.labels_
        rfm_scaled_df['Cluster'] = dbscan_model.labels_

        # Store in session state for other pages
        st.session_state['processed_df'] = processed_df
        st.session_state['rfm_df'] = rfm_df
        st.session_state['rfm_scaled_df'] = rfm_scaled_df
        st.session_state['scaler'] = scaler
        st.session_state['dbscan_model'] = dbscan_model
        st.session_state['user_label_map'] = USER_LABEL_MAP
        st.session_state['snapshot_date'] = snapshot_date

    st.sidebar.success("Data loaded, preprocessed, and clustered successfully!")

    # --- Cluster Label Mapping using User's Labels ---
    cluster_label_map = USER_LABEL_MAP
    rfm_df['Segment_Label'] = rfm_df['Cluster'].map(cluster_label_map)


    # --- Prediction Section in Main Area (DBSCAN Specific) ---
    st.header("Predict Customer Segment")
    st.write("Enter a Customer ID and optionally new transaction details to predict their segment.")

    customer_id_input = st.number_input("Enter Customer ID (e.g., 12345 or a new ID like 99999)", min_value=1, value=12347, step=1, key="customer_id_input")

    st.subheader("Optional: New Order Transaction Details")
    st.markdown("Fill these fields if you want to simulate a new order for an existing customer, or if this is a new customer's first order.")

    col_new1, col_new2, col_new3 = st.columns(3)
    with col_new1:
        new_invoice_no = st.text_input("InvoiceNo", value="", placeholder="e.g., NEW001", key="new_invoice_no")
        new_stock_code = st.text_input("StockCode", value="", placeholder="e.g., ITEM001", key="new_stock_code")
        new_description = st.text_input("Description", value="", placeholder="e.g., New Customer Item", key="new_description")
    with col_new2:
        new_quantity = st.number_input("Quantity", min_value=0, value=0, key="new_quantity") # Default to 0 for optional input
        new_price_per_item = st.number_input("Price (per item)", min_value=0.00, value=0.00, format="%.2f", key="new_price_per_item") # Default to 0.00
        new_invoice_date = st.date_input("InvoiceDate", value=dt.date.today(), key="new_invoice_date")
    with col_new3:
        new_country = st.text_input("Country", value="", placeholder="e.g., United Kingdom", key="new_country")
        st.markdown("*(Customer ID is entered above)*")

    # Single Predict Segment Button
    if st.button("Predict Segment", key="predict_segment_button"):
        # Determine if new transaction details were provided
        new_transaction_provided = (new_quantity > 0 and new_price_per_item > 0 and new_invoice_no != "")

        is_existing_customer = customer_id_input in rfm_df['Customer ID'].values

        current_predicted_recency = None
        current_predicted_orders = None
        current_predicted_price = None
        prediction_successful = False

        if is_existing_customer:
            st.success(f"Welcome back, Customer ID: {customer_id_input}!")
            existing_customer_data = rfm_df[rfm_df['Customer ID'] == customer_id_input].iloc[0]
            historical_recency = existing_customer_data['Recency']
            historical_orders = existing_customer_data['Orders']
            historical_price = existing_customer_data['Price']
            st.info(f"Historical RFM: Recency={historical_recency:.2f}, Orders={historical_orders:.2f}, Price=£{historical_price:.2f}")

            if new_transaction_provided:
                # Simulate new order for existing customer
                new_transaction_date_dt = dt.datetime.combine(new_invoice_date, dt.time.min)
                calculated_recency = max(0, (snapshot_date - new_transaction_date_dt).days) # Recency is 0 if new date is current/future
                calculated_orders = 1 # For this single new transaction
                calculated_price = new_quantity * new_price_per_item

                current_predicted_recency = calculated_recency
                current_predicted_orders = historical_orders + calculated_orders
                current_predicted_price = historical_price + calculated_price
                st.info(f"Updated RFM for new order: Recency={current_predicted_recency:.2f}, Orders={current_predicted_orders:.2f}, Price=£{current_predicted_price:.2f}")
                prediction_successful = True
            else:
                # Use historical data for existing customer if no new transaction
                current_predicted_recency = historical_recency
                current_predicted_orders = historical_orders
                current_predicted_price = historical_price
                st.info("No new transaction details provided. Predicting based on historical RFM.")
                prediction_successful = True

        else: # New Customer
            st.warning(f"Hello to the new customer (ID: {customer_id_input})!")
            if new_transaction_provided:
                # Calculate RFM for the new single transaction
                new_transaction_date_dt = dt.datetime.combine(new_invoice_date, dt.time.min)
                calculated_recency = max(0, (snapshot_date - new_transaction_date_dt).days) # Recency is 0 if new date is current/future
                calculated_orders = 1 # For a single new transaction
                calculated_price = new_quantity * new_price_per_item

                current_predicted_recency = calculated_recency
                current_predicted_orders = calculated_orders
                current_predicted_price = calculated_price
                st.info(f"Calculated RFM for new customer: Recency={calculated_recency}, Orders={calculated_orders}, Price=£{calculated_price:.2f}")
                prediction_successful = True
            else:
                st.error("As this is a new customer, please provide details for their first transaction to predict their segment.")
                prediction_successful = False

        # --- Display Prediction Results ---
        if prediction_successful:
            new_customer_data_for_prediction = pd.DataFrame([[current_predicted_recency, current_predicted_orders, current_predicted_price]],
                                             columns=['Recency', 'Orders', 'Price'])
            new_customer_scaled = scaler.transform(new_customer_data_for_prediction)

            if not rfm_scaled_df.empty:
                nn_model = NearestNeighbors(n_neighbors=1)
                nn_model.fit(rfm_scaled_df[['Recency_Scaled', 'Orders_Scaled', 'Price_Scaled']])

                distances, indices = nn_model.kneighbors(new_customer_scaled)
                closest_point_index_in_rfm_scaled_df = indices[0][0]

                predicted_cluster_num = rfm_scaled_df.loc[closest_point_index_in_rfm_scaled_df, 'Cluster']
                predicted_label = cluster_label_map.get(predicted_cluster_num, f"Cluster {predicted_cluster_num} (Unmapped)")

                st.subheader(f"Predicted Segment: {predicted_label}")
                st.markdown("---")
                st.write("Here's a summary of the characteristics for this predicted segment:")

                summary_predicted_cluster = rfm_df[rfm_df['Cluster'] == predicted_cluster_num][['Recency', 'Orders', 'Price']].mean()
                st.write(f"**Average Recency:** {summary_predicted_cluster['Recency']:.2f} days")
                st.write(f"**Average Orders:** {summary_predicted_cluster['Orders']:.2f} invoices")
                st.write(f"**Average Price:** £{summary_predicted_cluster['Price']:.2f}")
                st.write(f"**Number of customers in segment:** {len(rfm_df[rfm_df['Cluster'] == predicted_cluster_num])}")

                st.markdown(f"**Description of '{predicted_label}':**")
                if predicted_label == 'VIP Customers':
                    st.write("These customers are identified as VIPs. According to DBSCAN, these might be noise points or a distinct cluster depending on parameters. They are likely your most valuable customers, purchasing very recently, making many orders, and spending a significant amount (high price).")
                elif predicted_label == 'Churned Low Spenders':
                    st.write("These customers have not purchased recently, make few orders, and spend little (low price). They are likely churned and were not highly profitable. Focus on understanding why they left, but prioritize higher-value segments for re-engagement.")
                elif predicted_label == 'Engaged High Spenders':
                    st.write("These customers are highly engaged and contribute significantly to revenue. They make many orders and and spend well (high price). Nurture them to become VIPs.")
                elif predicted_label == 'Lost One-Time Buyers':
                    st.write("These customers made a purchase a long time ago and haven't returned, often having made only one order with low price. Re-engagement might be challenging but worth a try with specific campaigns.")
                elif predicted_label == 'New High-Value Buyers':
                    st.write("These are recent customers who have already shown good spending potential. They are important for growth; encourage repeat orders.")
                else:
                    st.write("No specific description available for this cluster. This might indicate an unusual customer profile or a need to refine cluster definitions.")

            else:
                st.warning("The dataset is empty or no clusters were formed. Please ensure your data file is correct and adjust 'eps' and 'min_samples' if necessary.")
                st.subheader("Predicted Segment: Undefined (No Data or Clusters Formed)")

    st.markdown("---")
    st.markdown("App developed using your provided code logic, adapted for DBSCAN and your specific cluster labels.")

if __name__ == "__main__":
    main()

