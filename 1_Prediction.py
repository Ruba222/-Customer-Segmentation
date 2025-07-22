# import streamlit as st
# import pandas as pd
# import numpy as np
# import datetime as dt
# from sklearn.preprocessing import RobustScaler
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns

# # Suppress warnings for cleaner output
# import warnings
# warnings.filterwarnings('ignore')

# # --- Configuration ---
# FILE_PATH = 'online_retail_II.xlsx'

# # --- User-defined Cluster Labels ---
# USER_LABEL_MAP = {
#     -1: 'VIP Customers',
#      0: 'Churned Low Spenders',
#      1: 'Engaged High Spenders',
#      2: 'Lost One-Time Buyers',
#      3: 'New High-Value Buyers'
# }

# # --- Data Loading and Preprocessing Function ---
# @st.cache_data
# def load_and_preprocess_data(file_path):
#     """
#     Loads the retail data and performs all necessary preprocessing steps.
#     """
#     try:
#         data = pd.read_excel(file_path)
#     except FileNotFoundError:
#         st.error(f"Error: The file '{file_path}' was not found. "
#                  "Please ensure it's in the same directory as the Streamlit app.")
#         return None

#     df = data.copy()

#     # 1. Drop rows with missing 'Description'
#     df.dropna(axis=0, how='any', subset=['Description'], inplace=True)

#     # 2. Remove cancelled transactions (Invoice starting with 'C')
#     df = df[~df['Invoice'].astype(str).str.startswith('C')]

#     # 3. Remove transactions with negative 'Quantity'
#     df = df[df['Quantity'] > 0]

#     # 4. Remove transactions with Invoice starting with 'A' and null Customer ID (bad debt adjustments)
#     df = df[~((df['Invoice'].astype(str).str.startswith('A')) & (df['Customer ID'].isnull()))]

#     # 5. Handle duplicates
#     df.drop_duplicates(inplace=True)
#     df.reset_index(drop=True, inplace=True)

#     # 6. Fill missing 'Customer ID' with 'Invoice' number and ensure numeric type
#     df['Customer ID'] = df['Customer ID'].fillna(df['Invoice']).astype(str)
#     df['Customer ID'] = pd.to_numeric(df['Customer ID'], errors='coerce')
#     df.dropna(subset=['Customer ID'], inplace=True)
#     df['Customer ID'] = df['Customer ID'].astype(int)

#     # 7. Calculate 'TotalPrice' for each transaction
#     df['TotalPrice'] = df['Quantity'] * df['Price']

#     return df

# # --- RFM Feature Engineering Function ---
# @st.cache_data
# def calculate_rfm(df):
#     """
#     Calculates Recency, Orders, and Price features for each customer.
#     """
#     snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

#     rfm_df = df.groupby('Customer ID').agg({
#         'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
#         'Invoice': 'nunique',
#         'TotalPrice': 'sum'
#     }).reset_index()

#     rfm_df.rename(columns={'InvoiceDate': 'Recency',
#                            'Invoice': 'Orders',
#                            'TotalPrice': 'Price'}, inplace=True)

#     rfm_df = rfm_df[rfm_df['Price'] > 0]
#     rfm_df = rfm_df[rfm_df['Orders'] > 0]

#     return rfm_df, snapshot_date

# # --- Clustering Function (DBSCAN) ---
# @st.cache_resource
# def train_dbscan_model(rfm_df_for_scaling, eps, min_samples):
#     """
#     Scales RFM features using RobustScaler and trains the DBSCAN clustering model.
#     Returns the scaler and the trained DBSCAN model.
#     """
#     scaler = RobustScaler()
#     rfm_scaled = scaler.fit_transform(rfm_df_for_scaling)
#     rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency_Scaled', 'Orders_Scaled', 'Price_Scaled'])

#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     dbscan.fit(rfm_scaled_df)

#     return scaler, dbscan, rfm_scaled_df

# # --- Main Application Logic for Prediction Page ---
# def main():
#     st.set_page_config(layout="wide", page_title="Customer Segmentation App (DBSCAN)")
#     st.image("assets/header.jpg", use_container_width=True)

#     st.title("🛍️ Customer Segmentation and Prediction (DBSCAN)")
#     st.markdown("This application helps you understand customer segments based on Recency, Orders and  Price analysis using **DBSCAN** clustering.")
#     st.markdown("DBSCAN identifies clusters based on density, and can also identify 'noise' points that don't belong to any cluster.")

#     # --- DBSCAN Parameters in Sidebar ---
#     st.sidebar.header("DBSCAN Parameters")
#     eps = 0.5  # Fixed value
#     min_samples = st.sidebar.slider("Min Samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.",
#                                     min_value=1, max_value=50, value=5, step=1)

#     # --- Data Loading and Model Training ---
#     st.sidebar.header("Application Status")
#     with st.spinner("Loading, preprocessing data, and training model... This may take a moment."):
#         processed_df = load_and_preprocess_data(FILE_PATH)

#         if processed_df is None:
#             st.stop()

#         rfm_df, snapshot_date = calculate_rfm(processed_df)
#         rfm_features = rfm_df[['Recency', 'Orders', 'Price']]
#         scaler, dbscan_model, rfm_scaled_df = train_dbscan_model(rfm_features, eps, min_samples)

#         # Add cluster labels to the RFM DataFrame
#         rfm_df['Cluster'] = dbscan_model.labels_
#         rfm_scaled_df['Cluster'] = dbscan_model.labels_

#         # Store in session state for other pages
#         st.session_state['processed_df'] = processed_df
#         st.session_state['rfm_df'] = rfm_df
#         st.session_state['rfm_scaled_df'] = rfm_scaled_df
#         st.session_state['scaler'] = scaler
#         st.session_state['dbscan_model'] = dbscan_model
#         st.session_state['user_label_map'] = USER_LABEL_MAP
#         st.session_state['snapshot_date'] = snapshot_date

#     st.sidebar.success("Data loaded, preprocessed, and clustered successfully!")

#     # --- Cluster Label Mapping using User's Labels ---
#     cluster_label_map = USER_LABEL_MAP
#     rfm_df['Segment_Label'] = rfm_df['Cluster'].map(cluster_label_map)

#     # --- Prediction Section in Main Area (DBSCAN Specific) ---
#     # [ ... Rest of the code remains unchanged ... ]

# if __name__ == "__main__":
#     main()








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
import warnings

# Suppress warnings for cleaner output
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
    try:
        data = pd.read_excel(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it's in the same directory as the Streamlit app.")
        return None

    df = data.copy()
    df.dropna(axis=0, how='any', subset=['Description'], inplace=True)
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df = df[~((df['Invoice'].astype(str).str.startswith('A')) & (df['Customer ID'].isnull()))]
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Customer ID'] = df['Customer ID'].fillna(df['Invoice']).astype(str)
    df['Customer ID'] = pd.to_numeric(df['Customer ID'], errors='coerce')
    df.dropna(subset=['Customer ID'], inplace=True)
    df['Customer ID'] = df['Customer ID'].astype(int)
    df['TotalPrice'] = df['Quantity'] * df['Price']
    return df

# --- RFM Feature Engineering Function ---
@st.cache_data
def calculate_rfm(df):
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
    scaler = RobustScaler()
    rfm_scaled = scaler.fit_transform(rfm_df_for_scaling)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency_Scaled', 'Orders_Scaled', 'Price_Scaled'])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(rfm_scaled_df)
    return scaler, dbscan, rfm_scaled_df

# --- Main App ---
def main():
    st.set_page_config(layout="wide", page_title="Customer Segmentation App (DBSCAN)")
    st.image("assets/header.jpg", use_container_width=True)

    st.title("🛍️ Customer Segmentation and Prediction (DBSCAN)")
    st.markdown("This app uses **DBSCAN** to segment customers by Recency, Orders, and Price.")

    # --- DBSCAN Parameters ---
    st.sidebar.header("DBSCAN Parameters")
    eps = 0.5
    st.sidebar.write(f"Using fixed epsilon (eps): {eps}")
    min_samples = st.sidebar.slider("Min Samples", min_value=1, max_value=50, value=5, step=1)

    # --- Load and Process Data ---
    st.sidebar.header("Application Status")
    with st.spinner("Processing data and training model..."):
        processed_df = load_and_preprocess_data(FILE_PATH)
        if processed_df is None:
            st.stop()

        rfm_df, snapshot_date = calculate_rfm(processed_df)
        rfm_features = rfm_df[['Recency', 'Orders', 'Price']]
        scaler, dbscan_model, rfm_scaled_df = train_dbscan_model(rfm_features, eps, min_samples)

        rfm_df['Cluster'] = dbscan_model.labels_
        rfm_scaled_df['Cluster'] = dbscan_model.labels_

        st.session_state['processed_df'] = processed_df
        st.session_state['rfm_df'] = rfm_df
        st.session_state['rfm_scaled_df'] = rfm_scaled_df
        st.session_state['scaler'] = scaler
        st.session_state['dbscan_model'] = dbscan_model
        st.session_state['user_label_map'] = USER_LABEL_MAP
        st.session_state['snapshot_date'] = snapshot_date

    st.sidebar.success("Clustering completed!")

    cluster_label_map = USER_LABEL_MAP
    rfm_df['Segment_Label'] = rfm_df['Cluster'].map(cluster_label_map)

    # --- Predict Customer Segment ---
    st.header("Predict Customer Segment")
    customer_id_input = st.number_input("Enter Customer ID", min_value=1, value=12347, step=1)

    st.subheader("Optional: New Order Transaction Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_invoice_no = st.text_input("InvoiceNo", value="")
        new_stock_code = st.text_input("StockCode", value="")
        new_description = st.text_input("Description", value="")
    with col2:
        new_quantity = st.number_input("Quantity", min_value=0, value=0)
        new_price_per_item = st.number_input("Price (per item)", min_value=0.00, value=0.00, format="%.2f")
        new_invoice_date = st.date_input("InvoiceDate", value=dt.date.today())
    with col3:
        new_country = st.text_input("Country", value="")
        st.markdown("*(Customer ID is entered above)*")

    if st.button("Predict Segment"):
        new_transaction_provided = (new_quantity > 0 and new_price_per_item > 0 and new_invoice_no != "")
        is_existing_customer = customer_id_input in rfm_df['Customer ID'].values

        current_predicted_recency = None
        current_predicted_orders = None
        current_predicted_price = None
        prediction_successful = False

        if is_existing_customer:
            st.success(f"Welcome back, Customer ID: {customer_id_input}!")
            existing_data = rfm_df[rfm_df['Customer ID'] == customer_id_input].iloc[0]
            r, o, p = existing_data['Recency'], existing_data['Orders'], existing_data['Price']
            st.info(f"Historical RFM: Recency={r:.2f}, Orders={o:.2f}, Price=£{p:.2f}")

            if new_transaction_provided:
                trans_date = dt.datetime.combine(new_invoice_date, dt.time.min)
                calculated_recency = max(0, (snapshot_date - trans_date).days)
                calculated_orders = 1
                calculated_price = new_quantity * new_price_per_item

                current_predicted_recency = calculated_recency
                current_predicted_orders = o + calculated_orders
                current_predicted_price = p + calculated_price
                prediction_successful = True
            else:
                current_predicted_recency = r
                current_predicted_orders = o
                current_predicted_price = p
                prediction_successful = True
        else:
            st.warning(f"New Customer (ID: {customer_id_input})")
            if new_transaction_provided:
                trans_date = dt.datetime.combine(new_invoice_date, dt.time.min)
                calculated_recency = max(0, (snapshot_date - trans_date).days)
                calculated_orders = 1
                calculated_price = new_quantity * new_price_per_item

                current_predicted_recency = calculated_recency
                current_predicted_orders = calculated_orders
                current_predicted_price = calculated_price
                prediction_successful = True
            else:
                st.error("Please provide transaction details for new customer.")
                prediction_successful = False

        if prediction_successful:
            new_df = pd.DataFrame([[current_predicted_recency, current_predicted_orders, current_predicted_price]],
                                  columns=['Recency', 'Orders', 'Price'])
            new_scaled = scaler.transform(new_df)

            if not rfm_scaled_df.empty:
                nn_model = NearestNeighbors(n_neighbors=1)
                nn_model.fit(rfm_scaled_df[['Recency_Scaled', 'Orders_Scaled', 'Price_Scaled']])
                distances, indices = nn_model.kneighbors(new_scaled)
                closest_idx = indices[0][0]
                predicted_cluster = rfm_scaled_df.loc[closest_idx, 'Cluster']
                predicted_label = cluster_label_map.get(predicted_cluster, f"Cluster {predicted_cluster} (Unmapped)")

                st.subheader(f"Predicted Segment: {predicted_label}")
                summary = rfm_df[rfm_df['Cluster'] == predicted_cluster][['Recency', 'Orders', 'Price']].mean()
                st.write(f"**Avg Recency:** {summary['Recency']:.2f} days")
                st.write(f"**Avg Orders:** {summary['Orders']:.2f}")
                st.write(f"**Avg Price:** £{summary['Price']:.2f}")
                st.write(f"**Customers in segment:** {len(rfm_df[rfm_df['Cluster'] == predicted_cluster])}")

                st.markdown(f"**Description of '{predicted_label}':**")
                if predicted_label == 'VIP Customers':
                    st.write("High-value customers. Very recent, frequent, and high spenders.")
                elif predicted_label == 'Churned Low Spenders':
                    st.write("Old, inactive, low-spending customers. Possibly lost.")
                elif predicted_label == 'Engaged High Spenders':
                    st.write("Active and valuable. Keep them engaged!")
                elif predicted_label == 'Lost One-Time Buyers':
                    st.write("Made one purchase a long time ago. Hard to re-engage.")
                elif predicted_label == 'New High-Value Buyers':
                    st.write("Recent customers with high spending. Nurture them!")
                else:
                    st.write("No description for this segment.")
            else:
                st.warning("No clusters formed. Check your dataset or adjust min_samples.")

    st.markdown("---")
    st.markdown("App developed with fixed epsilon (0.5) using DBSCAN clustering.")

if __name__ == "__main__":
    main()

