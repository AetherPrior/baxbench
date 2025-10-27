import streamlit as st
import pandas as pd
import json

# Metric definitions
METRICS = {
    'TMD': ('Threat Modeling', 'Explicitly names attacker goals; surfaces abuse cases before design decisions'),
    'DFL': ('Data Flow', 'Traces sensitive variables from source to sink; notes validation, transformation, storage'),
    'CFL': ('Security Control Flow generation', 'Traces logical order of security checks, ensuring no bypass paths exist'),
    'CWE': ('Common Weaknesses', 'Recalls likely bug classes (injection, XSS, etc.) and steers away from them'),
    'RCV': ('Recovery', 'Plans for code failure and adds isolation steps to contain damage'),
    'SCN': ('Security Constraints', 'Determines and enforces security constraints; sanitizes input with strict schemas'),
    'LEE': ('Least Exposure', 'Retains only necessary data; masks, tokenizes, or compartmentalizes sensitive fields'),
    'TST': ('Security Test Intent', 'Defines invariants and sketches tests to catch violations'),
    'ABU': ('Abuse - Resource Controls', 'Anticipates misuse; applies rate limits, quotas, timeouts'),
    'SCG': ('Scaffold Code Generation', 'Generates partial/skeletal code outlining structure before complete implementation')
}

# Page config
st.set_page_config(page_title="Security Annotation Tool", layout="wide")

# Load data (no caching - we want fresh data each session)
def load_data():
    return pd.read_csv('Untitled spreadsheet  deepseek_results_sampled.csv')

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = load_data()
    
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

if 'show_prompt' not in st.session_state:
    st.session_state.show_prompt = False

# Navigation
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    if st.button("← Previous", disabled=st.session_state.current_index == 0):
        st.session_state.current_index -= 1
        st.rerun()

with col2:
    st.markdown(f"<h3 style='text-align: center;'>Row {st.session_state.current_index + 1} of {len(st.session_state.df)}</h3>", unsafe_allow_html=True)

with col3:
    if st.button("Next →", disabled=st.session_state.current_index >= len(st.session_state.df) - 1):
        st.session_state.current_index += 1
        st.rerun()

# Jump to row
jump_to = st.number_input("Jump to row:", min_value=1, max_value=len(st.session_state.df), value=st.session_state.current_index + 1, key='jump')
if jump_to - 1 != st.session_state.current_index:
    st.session_state.current_index = jump_to - 1
    st.rerun()

st.divider()

# Current row data
row = st.session_state.df.iloc[st.session_state.current_index]

# Prompt toggle
if st.checkbox("Show Prompt", value=st.session_state.show_prompt):
    st.session_state.show_prompt = True
    with st.expander("Prompt", expanded=True):
        st.text(row['prompt'])
else:
    st.session_state.show_prompt = False

# Reasoning trace
st.subheader("Reasoning Trace")
st.text_area("", value=row['gen_text'], height=300, disabled=True, label_visibility="collapsed")

st.divider()

# Annotations
st.subheader("Annotations")

# Track changes
changes_made = False

for metric_code, (metric_name, metric_desc) in METRICS.items():
    st.markdown(f"### {metric_code}: {metric_name}")
    st.caption(metric_desc)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # GPT annotations
        presence = row.get(f'presence_{metric_code}_{metric_name}', 'N/A')
        quality = row.get(f'quality_{metric_code}_{metric_name}', 'N/A')
        evidence = row.get(f'evidence_lines_{metric_code}_{metric_name}', 'N/A')
        rationale = row.get(f'rationale_{metric_code}_{metric_name}', 'N/A')
        
        st.markdown(f"**GPT Presence:** {presence}")
        st.markdown(f"**GPT Quality:** {quality}")
        st.markdown(f"**GPT Evidence Lines:** {evidence}")
        st.markdown(f"**GPT Rationale:** {rationale}")
    
    with col2:
        # Human annotation
        human_col = f'human_annotation_{metric_code}'
        current_value = row.get(human_col, '')
        
        # Determine current selection
        if pd.isna(current_value) or current_value == '':
            default_idx = 0
        elif current_value == 'A':
            default_idx = 1
        elif current_value == 'D':
            default_idx = 2
        else:
            default_idx = 3
        
        annotation = st.radio(
            "Your annotation:",
            options=['Not annotated', 'A (Agree)', 'D (Disagree)', 'Other'],
            index=default_idx,
            key=f'radio_{metric_code}_{st.session_state.current_index}'
        )
        
        other_text = None
        if annotation == 'Other':
            other_text = st.text_input(
                "Explain:",
                value=current_value if default_idx == 3 else '',
                key=f'other_{metric_code}_{st.session_state.current_index}'
            )
        
        # Update dataframe if changed
        new_value = ''
        if annotation == 'A (Agree)':
            new_value = 'A'
        elif annotation == 'D (Disagree)':
            new_value = 'D'
        elif annotation == 'Other' and other_text:
            new_value = other_text
        
        if new_value != current_value:
            st.session_state.df.at[st.session_state.current_index, human_col] = new_value
            changes_made = True
    
    st.divider()

# Save button
if st.button("💾 Save Progress", type="primary"):
    st.session_state.df.to_csv('Untitled spreadsheet  deepseek_results_sampled.csv', index=False)
    st.success("✅ Saved!")

# Download button
csv = st.session_state.df.to_csv(index=False)
st.download_button(
    label="📥 Download Annotated CSV",
    data=csv,
    file_name="annotated_results.csv",
    mime="text/csv"
)

# Progress indicator
annotated_count = 0
for metric_code in METRICS.keys():
    human_col = f'human_annotation_{metric_code}'
    if not pd.isna(row.get(human_col, '')) and row.get(human_col, '') != '':
        annotated_count += 1

st.sidebar.metric("Progress for this row", f"{annotated_count}/10")

# Overall progress
total_annotations = 0
for idx in range(len(st.session_state.df)):
    for metric_code in METRICS.keys():
        human_col = f'human_annotation_{metric_code}'
        val = st.session_state.df.iloc[idx].get(human_col, '')
        if not pd.isna(val) and val != '':
            total_annotations += 1

total_possible = len(st.session_state.df) * 10
st.sidebar.metric("Overall Progress", f"{total_annotations}/{total_possible} ({100*total_annotations//total_possible}%)")