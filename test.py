import pandas as pd
import streamlit as st

db_path = "welddb/welddb.data"
# Read the space-separated file
df = pd.read_csv(db_path, sep="\s+", header=None)


st.write(df)
