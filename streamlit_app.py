import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
fuel = st.sidebar.selectbox("Fuel Type", sorted(df['fuel'].dropna().unique()))
seller = st.sidebar.selectbox("Seller Type", sorted(df['seller_type'].dropna().unique()))
transmission = st.sidebar.selectbox("Transmission", sorted(df['transmission'].dropna().unique()))
owner = st.sidebar.selectbox("Owner", sorted(df['owner'].dropna().unique()))
year = st.sidebar.selectbox("Manufacturing Year", sorted(df['year'].dropna().unique(), reverse=True))

# Filter the data
filtered_df = df[
    (df['fuel'] == fuel) &
    (df['seller_type'] == seller) &
    (df['transmission'] == transmission) &
    (df['owner'] == owner) &
    (df['year'] == year)
]

# Show filtered chart
st.title("🚗 Car Price Explorer")
st.subheader(f"Price Distribution for Selected Filters")

fig, ax = plt.subplots()
ax.hist(filtered_df['selling_price'], bins=20, color='skyblue', edgecolor='black')
ax.set_xlabel("Selling Price")
ax.set_ylabel("Number of Cars")
st.pyplot(fig)

# Show data summary
st.markdown("### 📊 Summary Stats")
st.write(filtered_df['selling_price'].describe())

# Show data table if checked
if st.checkbox("Show data table"):
    st.dataframe(filtered_df)
