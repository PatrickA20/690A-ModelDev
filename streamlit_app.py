import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

df = load_data()

# Title and description
st.title("🚗 Car Price Explorer")
st.write("Explore car selling prices by fuel type and seller type using real data from Car Dekho.")

# Sidebar filters
st.sidebar.header("Filter Options")
fuel = st.sidebar.selectbox("Choose Fuel Type", df['fuel'].unique())
seller = st.sidebar.selectbox("Choose Seller Type", df['seller_type'].unique())

# Filtered data
filtered = df[(df['fuel'] == fuel) & (df['seller_type'] == seller)]

# Display filtered results
st.subheader(f"Selling Price Distribution for {fuel} Cars Sold by {seller}s")

# Plot
fig, ax = plt.subplots()
ax.hist(filtered['selling_price'], bins=20, color='skyblue', edgecolor='black')
ax.set_xlabel("Selling Price")
ax.set_ylabel("Number of Cars")
st.pyplot(fig)

# Show summary stats
st.markdown("### Summary Statistics")
st.write(filtered['selling_price'].describe())
