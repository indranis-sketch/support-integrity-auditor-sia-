import streamlit as st
import pandas as pd
import json
from src.pseudo_label import generate_pseudo_labels
from src.inference import generate_evidence_dossier
import seaborn as sns
import matplotlib.pyplot as plt

def process_batch(df):
    df = generate_pseudo_labels(df)
    mismatches = df[df['mismatch_label'] == 1]
    
    dossiers = []
    for _, row in mismatches.iterrows():
        dossiers.append(generate_evidence_dossier(row))
        
    return df, dossiers

def main():
    st.set_page_config(page_title="Support Integrity Auditor", layout="wide")
    st.title("Support Integrity Auditor (SIA)")
    st.write("Detecting Priority Mismatch in support tickets.")
    
    tab1, tab2 = st.tabs(["Single Ticket Analysis", "Batch Processing & Dashboard"])
    
    with tab1:
        st.header("Single Ticket Analysis")
        with st.form("single_ticket_form"):
            col1, col2 = st.columns(2)
            subject = col1.text_input("Ticket Subject")
            desc = st.text_area("Ticket Description")
            priority = col1.selectbox("Assigned Priority", ["Low", "Medium", "High", "Critical"])
            channel = col2.selectbox("Channel", ["email", "chat", "phone", "social media"])
            res_time = col2.number_input("Resolution Time (hrs)", min_value=0.0, value=24.0)
            
            submit = st.form_submit_button("Analyze Ticket")
            
        if submit:
            ticket_data = pd.DataFrame([{
                'Ticket ID': 'single_1',
                'Ticket Subject': subject,
                'Ticket Description': desc,
                'Ticket Priority': priority,
                'Ticket Channel': channel,
                'Resolution Time': res_time
            }])
            
            processed_df, dossiers = process_batch(ticket_data)
            
            st.subheader("Analysis Results")
            if dossiers:
                st.error(f"Priority Mismatch Detected: {dossiers[0]['mismatch_type']}")
                st.json(dossiers[0])
            else:
                st.success("Ticket Priority appears Consistent.")
                
    with tab2:
        st.header("Batch Processing & Dashboard")
        uploaded_file = st.file_uploader("Upload CRM Dataset (CSV)", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} tickets.")
            
            if st.button("Run Auditor Pipeline"):
                with st.spinner("Running pipeline..."):
                    processed_df, dossiers = process_batch(df)
                
                st.success(f"Pipeline completed. Found {len(dossiers)} mismatches.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Mismatch Types")
                    if dossiers:
                        types = [d['mismatch_type'] for d in dossiers]
                        st.bar_chart(pd.Series(types).value_counts())
                
                with col2:
                    st.subheader("Severity Delta Heatmap")
                    pivot = processed_df.pivot_table(index='Ticket Priority', columns='inferred_severity', values='Ticket ID', aggfunc='count', fill_value=0)
                    fig, ax = plt.subplots()
                    sns.heatmap(pivot, annot=True, cmap="YlOrRd", ax=ax, fmt="d")
                    st.pyplot(fig)
                    
                st.subheader("Mismatch Dossiers")
                st.json(dossiers)

if __name__ == "__main__":
    main()
