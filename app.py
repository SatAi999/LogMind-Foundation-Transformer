import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import torch
from infer import LogMindInference

# Page Configuration - Premium Dark Theme
st.set_page_config(
    page_title="LogMind - Enterprise Log Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    .main {
        background-color: #0F111A;
        color: #F0F2F6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #1A1D2B;
        padding: 8px 12px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: #1A1D2B;
        border-radius: 4px;
        color: #8E9AAF;
        font-weight: 600;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4C6EF5 !important;
        color: white !important;
    }
    .card {
        background-color: #1A1D2B;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #2D3142;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #4C6EF5;
    }
    .metric-label {
        font-size: 14px;
        color: #8E9AAF;
    }
</style>
""", unsafe_value_path=True)

# Helper to load inference backend (cached to avoid reloading models on click)
@st.cache_resource
def get_inference_backend():
    backend = LogMindInference()
    # Build database of validation incidents (limit max lines to keep it fast)
    backend.build_similarity_database(
        log_path=backend.config["data"]["log_path"],
        label_path=backend.config["data"]["label_path"],
        max_lines=20000
    )
    return backend

# App header
st.title("🧠 LogMind")
st.subheader("A Foundation Transformer Architecture for Enterprise Log Intelligence")
st.markdown("---")

# Load backend
with st.spinner("Initializing LogMind Transformer and indexing similarity database..."):
    try:
        backend = get_inference_backend()
        st.sidebar.success("LogMind Model Loaded (CUDA active)" if torch.cuda.is_available() else "LogMind Model Loaded (CPU active)")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Sidebar - Diagnostics and Hyperparameters
st.sidebar.header("Architecture Parameters")
st.sidebar.info(f"""
- **Shared Encoder Stack**: Manual Transformer
- **Embedding Dim (d_model)**: {backend.config['model']['d_model']}
- **Attention Heads**: {backend.config['model']['n_heads']}
- **Encoder Layers**: {backend.config['model']['n_layers']}
- **Hidden Dim (d_ff)**: {backend.config['model']['d_ff']}
- **Vocabulary Size**: {len(backend.tokenizer)} templates
- **Max Sequence Length**: {backend.config['data']['max_len']}
""")

# Setup Navigation Tabs
tab_home, tab_infer, tab_sim, tab_gen = st.tabs([
    "📊 Dashboard Home & Analytics",
    "🔍 Interactive Multi-Task Inference",
    "🚀 Incident Similarity Search",
    "✍️ Autoregressive Log Generation"
])

# Tab 1: Home & Analytics
with tab_home:
    st.markdown("### Foundation Model Lifecycle Analytics")
    
    # Load final test metrics
    metrics_path = "checkpoints/test_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Failure Accuracy", f"{metrics['failure_prediction']['accuracy']*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Failure Precision", f"{metrics['failure_prediction']['precision']*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Failure Recall", f"{metrics['failure_prediction']['recall']*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Failure F1-Score", f"{metrics['failure_prediction']['f1']*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col5:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("RCA Macro F1", f"{metrics['root_cause_analysis']['f1_macro']*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
    # Display analytics plots
    st.markdown("#### Experiment Tracking & Visualization Plots")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if os.path.exists("plots/pretraining_curves.png"):
            st.image("plots/pretraining_curves.png", caption="Stage 1: Self-Supervised MLM Pretraining curves", use_container_width=True)
        if os.path.exists("plots/anomaly_timeline.png"):
            st.image("plots/anomaly_timeline.png", caption="Anomaly Prediction Timeline & Thresholds", use_container_width=True)
            
    with col_p2:
        if os.path.exists("plots/finetuning_curves.png"):
            st.image("plots/finetuning_curves.png", caption="Stage 2: Multi-Task Supervised Fine-Tuning curves", use_container_width=True)
        if os.path.exists("plots/test_embeddings_tsne.png"):
            st.image("plots/test_embeddings_tsne.png", caption="t-SNE clusters of Sequence Embeddings by RCA category", use_container_width=True)

# Tab 2: Interactive Inference
with tab_infer:
    st.markdown("### Live Multi-Task Sequence Inference")
    st.write("Enter a sequence of raw log lines or select one of the typical enterprise incident templates below.")
    
    # Pre-coded templates for quick demo
    sample_templates = {
        "Custom Raw Input": "",
        "Normal Operation Sequence": (
            "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010\n"
            "081109 203518 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/job_200811092030_0001/job.jar. blk_-1608999687919862906\n"
            "081109 203519 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.10.6:40524 dest: /10.250.10.6:50010\n"
            "081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687919862906 terminating\n"
            "081109 203519 145 INFO dfs.DataNode$PacketResponder: Received block blk_-1608999687919862906 of size 91178 from /10.250.10.6\n"
            "081109 203519 29 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.10.6:50010 is added to blk_-1608999687919862906 size 91178"
        ),
        "DataNode Write Exception (WriteFailure)": (
            "081109 203615 142 INFO dfs.DataNode$DataXceiver: Receiving block blk_-3544583377289625738 src: /10.250.19.102:39325 dest: /10.250.19.102:50010\n"
            "081109 203615 26 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/job_200811092030_0001/job.split. blk_-3544583377289625738\n"
            "081109 203616 142 WARN dfs.DataNode$DataXceiver: writeBlock blk_-3544583377289625738 received exception java.io.IOException: Could not read from stream\n"
            "081109 203616 142 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-3544583377289625738 terminating"
        ),
        "Network Timeout Exception (ConnectionTimeout)": (
            "081110 143210 142 INFO dfs.DataNode$DataXceiver: Receiving block blk_-8531310335568756456 src: /10.250.10.6:54200 dest: /10.250.10.6:50010\n"
            "081110 143211 26 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/job_200811092030_0001/job.jar. blk_-8531310335568756456\n"
            "081110 143215 142 WARN dfs.DataNode$PacketResponder: PacketResponder blk_-8531310335568756456 1 Exception java.net.SocketTimeoutException: 60000 millis timeout left\n"
            "081110 143215 142 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-8531310335568756456 terminating"
        ),
        "DataNode Serving Failure (ServingFailure)": (
            "081110 153010 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_1111111111111111 src: /10.250.10.6:54200 dest: /10.250.10.6:50010\n"
            "081110 153011 143 WARN dfs.DataNode$DataXceiver: /10.250.19.102:50010:Got exception while serving blk_1111111111111111 to /10.250.19.102\n"
            "081110 153012 143 INFO dfs.DataNode$DataXceiver: writeBlock blk_1111111111111111 received exception java.io.IOException: Broken pipe"
        )
    }
    
    selected_demo = st.selectbox("Quick Demo Templates", list(sample_templates.keys()))
    default_text = sample_templates[selected_demo]
    
    log_input = st.text_area("Raw Log Sequence Input", value=default_text, height=180)
    
    if st.button("Run Multi-Task Analysis", type="primary"):
        lines = [line.strip() for line in log_input.split("\n") if line.strip()]
        if len(lines) == 0:
            st.warning("Please enter at least one log line.")
        else:
            with st.spinner("Executing Shared Transformer Encoder..."):
                res = backend.predict_sequence(lines)
                
            col_res1, col_res2 = st.columns([1, 1])
            with col_res1:
                st.markdown("#### Model Predictions")
                
                # Anomaly Meter
                prob = res["anomaly_probability"]
                st.metric("Anomaly Probability Score", f"{prob*100:.2f}%")
                if prob > 0.5:
                    st.error("🚨 ANOMALOUS SEQUENCE DETECTED")
                else:
                    st.success("✅ NORMAL SEQUENCE")
                    
                # RCA Prediction
                st.metric("Predicted Root Cause", res["predicted_rca"])
                
                # Probabilities table
                st.write("**Root Cause Class Probabilities:**")
                for cat, p_val in res["rca_probabilities"].items():
                    st.progress(p_val, text=f"{cat}: {p_val*100:.2f}%")
                    
            with col_res2:
                st.markdown("#### Extracted Event Templates")
                for i, temp in enumerate(res["templates"]):
                    st.info(f"Step {i+1}: `{temp}`")
                    
            # Plt Attention Map
            st.markdown("#### Low-Level Layer 3 Self-Attention Map (Head 0)")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                res["attention_map"][0][:len(res["valid_tokens"]), :len(res["valid_tokens"])],
                xticklabels=res["valid_tokens"],
                yticklabels=res["valid_tokens"],
                annot=True,
                fmt=".2f",
                cmap="viridis",
                ax=ax
            )
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            plt.close(fig)

# Tab 3: Incident Similarity Search
with tab_sim:
    st.markdown("### Vector-Similarity Incident Search")
    st.write("Given a query log sequence, query the index database of incidents using cosine similarity over the learned contrastive embeddings.")
    
    if not backend.indexed_sequences:
        st.warning("No similarity database available.")
    else:
        # User chooses a query sequence from database anomalies
        anomaly_indices = [i for i, lbl in enumerate(backend.indexed_rca_labels) if lbl != 0]
        
        # Format list to show in selectbox
        query_options = {
            f"Incident {i} | Category: {backend.parser.rca_keywords.get(backend.indexed_rca_labels[idx], ['OtherAnomaly'])[0]} | Prompt: {backend.indexed_sequences[idx][0][:60]}...": idx
            for i, idx in enumerate(anomaly_indices[:20]) # Show first 20 anomaly incidents
        }
        
        selected_query = st.selectbox("Select Query Incident Sequence", list(query_options.keys()))
        query_idx = query_options[selected_query]
        
        query_seq = backend.indexed_sequences[query_idx]
        query_rca = backend.indexed_rca_labels[query_idx]
        
        st.write("**Query Sequence Templates:**")
        for i, temp in enumerate(query_seq):
            st.code(f"Step {i+1}: {temp}")
            
        if st.button("Search Similar Incidents in Database", type="primary"):
            # Compute query embedding
            token_ids = backend.tokenizer.encode(query_seq, add_special_tokens=True, max_len=backend.config["data"]["max_len"])
            input_tensor = torch.tensor([token_ids]).to(backend.device)
            padding_mask = (input_tensor != backend.tokenizer.pad_id).to(backend.device)
            
            with torch.no_grad():
                outputs = backend.model(input_tensor, padding_mask, mask_type="bidirectional", run_heads=["contrastive"])
                query_emb = outputs["embeddings"][0].cpu().numpy()
                
            # Perform similarity search
            results = backend.search_similar_incidents(query_emb, top_k=4)
            
            st.markdown("#### Search Results (Top 4 closest matches)")
            
            for rank, r in enumerate(results):
                st.markdown(f"""
                <div style="background-color: #1E2235; padding: 15px; border-radius: 8px; margin-bottom: 12px; border-left: 5px solid #4C6EF5;">
                    <strong>Rank {rank+1} | Cosine Similarity Score: {r['similarity']:.4f} | Label: {r['rca_category']}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("Show matched sequence templates"):
                    for i, step in enumerate(r["sequence"]):
                        st.write(f"Step {i+1}: `{step}`")

# Tab 4: Autoregressive Generation
with tab_gen:
    st.markdown("### Autoregressive Log Event Generation")
    st.write("LogMind can generate sequence paths by utilizing the Causal Attention Mask to predict the next event template autoregressively.")
    
    prompt_options = {
        "Allocation Prompt": [
            "BLOCK* NameSystem.allocateBlock: <path><num>. <block_id>",
            "Receiving block <block_id> src: <ip> dest: <ip>"
        ],
        "Verification Prompt": [
            "Receiving block <block_id> src: <ip> dest: <ip>",
            "Received block <block_id> of size <num> from /<ip>",
            "Verification succeeded for <block_id>"
        ],
        "Error Trigger Prompt": [
            "Receiving block <block_id> src: <ip> dest: <ip>",
            "writeBlock <block_id> received exception java.io.IOException: Could not read from stream"
        ]
    }
    
    selected_prompt_key = st.selectbox("Select Seed Prompt", list(prompt_options.keys()))
    seed_prompt = prompt_options[selected_prompt_key]
    
    st.write("**Seed Prompt Events:**")
    for i, step in enumerate(seed_prompt):
        st.code(f"Step {i+1}: {step}")
        
    col_gen1, col_gen2 = st.columns(2)
    with col_gen1:
        gen_len = st.slider("Max Generation Length", min_value=1, max_value=20, value=8)
    with col_gen2:
        temp_val = st.slider("Sampling Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.1)
        
    if st.button("Generate Log Path", type="primary"):
        with st.spinner("Autoregressive decoding with causal mask active..."):
            generated_seq = backend.generate_next_events(seed_prompt, max_gen_len=gen_len, temperature=temp_val)
            
        st.markdown("#### Generated Sequence Path")
        
        # Display with step-by-step styling
        for i, step in enumerate(generated_seq):
            if i < len(seed_prompt):
                st.write(f"Step {i+1} (Seed): `{step}`")
            else:
                st.markdown(f"**Step {i+1} (Generated):** <code style='color: #4C6EF5;'>{step}</code>", unsafe_allow_html=True)
