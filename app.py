#!/usr/bin/env python3

import os
import streamlit as st
st.title("YO")
st.write("this is the submission ig welcome")


folder_path = "/tmp/models"
file_name = "xgb_brain_tumor.pkl"
full_path = os.path.join(folder_path, file_name)
if os.path.exists(full_path):
    st.success(f"{file_name} is present in {folder_path}")
else:
    st.error(f"{file_name} not found in {folder_path}")
