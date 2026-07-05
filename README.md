# LogMind: A Unified Foundation Transformer for Enterprise Log Intelligence

LogMind is a custom, first-principles Transformer Encoder architecture designed for multi-task enterprise log intelligence. Built using low-level PyTorch tensor operations, it implements the complete Transformer Encoder stack from scratch. The system is designed to perform self-supervised pretraining (Masked Language Modeling and Causal Next-Event prediction) over raw log sequences and then transfer those weights to execute concurrent downstream tasks: anomaly detection, root cause classification, and vector-similarity incident search.

---

## 🏗️ Architectural Overview & Dual-Stage Lifecycle

LogMind structured log intelligence as a two-stage **pretrain-then-finetune** lifecycle, mirroring modern foundation model paradigms:

```
========================================================================
STAGE 1: SELF-SUPERVISED PRETRAINING (Representation Learning)
========================================================================
[Raw Logs] -> [LogParser] -> [Tokenizer] -> [MLM / CLM Dataset]
                                                  |
                                                  v
                                    +---------------------------+
                                    | Shared Transformer Encoder|
                                    +---------------------------+
                                                  |
                         +------------------------+------------------------+
                         | (Bidirectional Mask)                            | (Causal Mask)
                         v                                                 v
           +---------------------------+                     +---------------------------+
           |   Masked Token Predictor  |                     |    Next Event Generator   |
           +---------------------------+                     +---------------------------+

========================================================================
STAGE 2: MULTI-TASK SUPERVISED FINE-TUNING (Downstream Operations)
========================================================================
[Pretrained Encoder Weights]
            |
            v
+---------------------------+
| Shared Transformer Encoder|
+---------------------------+
            |
            +-----------> [CLS Pooling] -> Sequence Representation
                                |
         +----------------------+----------------------+
         |                      |                      |
         v                      v                      v
+------------------+   +------------------+   +------------------+
| Anomaly Detector |   |     RCA Head     |   | Contrastive Head |
| (Binary Failure) |   | (6-Class Failure)|   | (L2 Normalized)  |
+------------------+   +------------------+   +------------------+
                                                       |
                                                       v
                                              [Vector Search Index]
```

---

## 🪵 Log Preprocessing & Normalization Pipeline

Raw enterprise logs contain highly volatile parameters (such as IP addresses, timestamps, ports, and memory addresses) that lead to an infinite vocabulary. LogMind uses a deterministic regular-expression-based **LogParser** to normalize these fields into placeholders, collapsing the log stream into a structured set of event templates.

### 1. Raw Log Structure
HDFS log lines follow the format:
```
081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
```
* **Date**: `081109` (YYMMDD)
* **Time**: `203518` (HHMMSS)
* **Thread ID**: `143`
* **Log Level**: `INFO` (INFO, WARN, ERROR, FATAL, DEBUG)
* **Component**: `dfs.DataNode$DataXceiver`
* **Message**: `Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010`

### 2. Regex Normalization Dictionary
The message is parsed and normalized using the following sequence of regex templates:
| Source Pattern | Regex Pattern | Placeholder |
| :--- | :--- | :--- |
| Block ID | `blk_[-]?\d+` | `<block_id>` |
| IP with Port | `/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+` | `<ip>` |
| Standard IP | `\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}` | `<ip>` |
| Numbers | `\b\d+\b` | `<num>` |
| File Paths | `/[a-zA-Z0-9_\-\./]+` | `<path>` |
| Hex Addresses | `0x[0-9a-fA-F]+` | `<hex>` |

*Example output template:*
`Receiving block <block_id> src: <ip> dest: <ip>`

### 3. Vocabulary & Special Tokens
The **LogTokenizer** maps these templates to integer IDs. It reserves the first five IDs for special tokens:
* `[PAD]` (ID `0`): Used to pad sequences to a uniform length.
* `[UNK]` (ID `1`): Replaces out-of-vocabulary event templates.
* `[CLS]` (ID `2`): Prepended to every sequence. The final hidden state of `[CLS]` represents the aggregated sequence representation.
* `[SEP]` (ID `3`): Appended to the end of every sequence.
* `[MASK]` (ID `4`): Replaces tokens selected for pretraining masking.

---

## 🧮 Custom Transformer Architecture: First-Principles Math

The entire Transformer stack is custom-implemented in `logmind/models/`. Below are the mathematical formulations of the custom modules:

### 1. Token Embeddings & Positional Encodings
The input token IDs are projected to a dense representation:
$$E_{token} = W_{vocab}[X_{input}] \quad \text{where } W_{vocab} \in \mathbb{R}^{\text{vocab\_size} \times d_{model}}$$
Weights are initialized with a standard normal distribution scaled by $0.02$.

We support two modes of positional encodings:
* **Sinusoidal Positional Encodings**: Non-learnable coordinates representing trigonometric frequencies:
  $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
* **Learned Positional Embeddings**: A parameter weight matrix $W_{pos} \in \mathbb{R}^{\text{max\_seq\_len} \times d_{model}}$ optimized during backpropagation.

The final input representation is:
$$X_0 = \text{Dropout}(E_{token} + PE)$$

### 2. Multi-Head Self-Attention (MHA)
Inputs are projected to Query ($Q$), Key ($K$), and Value ($V$) matrices using manual parameter multiplications:
$$Q = X W_q + b_q, \quad K = X W_k + b_k, \quad V = X W_v + b_v \quad \text{where } W_q, W_k, W_v \in \mathbb{R}^{d_{model} \times d_{model}}$$

These projections are reshaped into $H$ heads of dimension $d_k = d_{model} / H$:
$$q, k, v \in \mathbb{R}^{\text{batch\_size} \times H \times \text{seq\_len} \times d_k}$$

The attention weights are computed using a scaled dot-product:
$$\text{Attention}(q, k, v) = \text{softmax}\left(\frac{q k^T}{\sqrt{d_k}} + M\right) v$$
* The scaling factor $\frac{1}{\sqrt{d_k}}$ prevents the dot products from growing excessively large in high dimensions, which would push the softmax function into regions with vanishing gradients.
* **Modular Masking Matrix ($M$)**:
  * *Bidirectional Masking*: Ensures that padding tokens do not receive attention:
    $$M_{i,j} = \begin{cases} 0 & \text{if } j \text{ is active} \\ -10^9 & \text{if } j \text{ is padding} \end{cases}$$
  * *Causal Masking*: Combines the padding mask with a lower-triangular causal matrix to enforce autoregressive restrictions (token $i$ cannot attend to $j > i$):
    $$M_{i,j} = \begin{cases} 0 & \text{if } j \le i \text{ and } j \text{ is active} \\ -10^9 & \text{if } j > i \text{ or } j \text{ is padding} \end{cases}$$

### 3. Layer Normalization
To stabilize activation distributions, LayerNorm normalizes each sample across the final feature dimension:
$$\text{LN}(X) = \gamma \odot \left(\frac{X - \mu}{\sqrt{\sigma^2 + \epsilon}}\right) + \beta$$
* Mean ($\mu$): $\mu = \frac{1}{D}\sum_{i=1}^D X_i$
* Biased Variance ($\sigma^2$): $\sigma^2 = \frac{1}{D}\sum_{i=1}^D (X_i - \mu)^2$
* $\gamma$ and $\beta$ are learnable parameters initialized to $1.0$ and $0.0$, respectively. $\epsilon = 10^{-5}$ prevents division by zero.

### 4. Residual Routing (Pre-LN vs. Post-LN)
* **Pre-LN (Default)**: Normalization is applied to the input *before* the sublayer, and the output is added to the residual. This preserves an identity gradient path, allowing deep encoders to train stably.
  $$X_{intermediate} = X_{in} + \text{Dropout}(\text{Sublayer}(\text{LN}(X_{in})))$$
* **Post-LN**: Normalization is applied *after* adding the residual:
  $$X_{intermediate} = \text{LN}(X_{in} + \text{Dropout}(\text{Sublayer}(X_{in})))$$

---

## 🎯 Multi-Task Downstream Prediction Heads

LogMind maps the final encoder representations to task-specific heads:

### 1. Next Log Event & MLM Head
Projects the final sequence representations back to the vocabulary space to predict token probabilities:
$$\text{Logits}_{MLM} = H_{encoder} W_{vocab\_proj} + b_{vocab\_proj} \quad \text{where } H_{encoder} \in \mathbb{R}^{B \times L \times d_{model}}$$

### 2. Anomaly Classification (Failure Prediction) Head
Performs sequence classification using a two-layer Multi-Layer Perceptron (MLP) over the `[CLS]` token representation ($h_{CLS} \in \mathbb{R}^{d_{model}}$):
$$h_{intermediate} = \tanh(h_{CLS} W_{anom1} + b_{anom1})$$
$$\text{Logit}_{anomaly} = h_{intermediate} W_{anom2} + b_{anom2} \quad \text{where } \text{Logit}_{anomaly} \in \mathbb{R}^1$$

### 3. Root Cause Analysis (RCA) Head
Categorizes sequences into 6 distinct classes (0: Normal, 1: WriteFailure, 2: ConnectionTimeout, 3: ServingFailure, 4: ReplicaVolumeError, 5: OtherAnomaly):
$$\text{Logits}_{RCA} = \text{GELU}(h_{CLS} W_{rca1} + b_{rca1}) W_{rca2} + b_{rca2} \quad \text{where } \text{Logits}_{RCA} \in \mathbb{R}^6$$

### 4. Contrastive Projection Head
Projects the sequence embedding to a lower-dimensional metric space and applies $L_2$ normalization:
$$z = h_{CLS} W_{proj} + b_{proj} \quad \text{where } z \in \mathbb{R}^{d_{emb}}$$
$$e = \frac{z}{\|z\|_2}$$
This guarantees that the dot product between two embeddings is exactly their cosine similarity.

---

## ⚡ Supervised Multi-Task Loss Functions

During training, the joint optimization loss is computed dynamically based on the model's active mode:

### 1. Stage 1: Self-Supervised Losses
* **MLM Loss**: Standard cross-entropy loss over masked positions (unmasked positions are set to $-100$ to exclude them from the gradient calculation):
  $$\mathcal{L}_{MLM} = -\frac{1}{N_{masked}} \sum_{i \in \text{masked}} \log P(x_i = y_i)$$
* **CLM Loss**: Shifts targets by 1 step to predict subsequent event templates:
  $$\mathcal{L}_{CLM} = -\frac{1}{L-1} \sum_{t=0}^{L-2} \log P(x_{t+1} | x_{\le t})$$

### 2. Stage 2: Joint Supervised Fine-Tuning Losses
$$\mathcal{L}_{total} = w_1 \mathcal{L}_{anomaly} + w_2 \mathcal{L}_{RCA} + w_3 \mathcal{L}_{contrastive}$$
* **$\mathcal{L}_{anomaly}$**: Binary Cross-Entropy with Logits.
* **$\mathcal{L}_{RCA}$**: Multi-class Cross-Entropy.
* **$\mathcal{L}_{contrastive}$ (Siamese Pairwise Contrastive Loss)**: Evaluates pairwise similarity $s_{i,j} = e_i \cdot e_j$ for a batch of size $B$:
  $$\mathcal{L}_{contrastive} = \frac{1}{|P|} \sum_{(i,j) \in P} (1 - s_{i,j}) + \frac{1}{|N|} \sum_{(i,j) \in N} \max(0, s_{i,j} - m)^2$$
  * $P$: Positive pairs where labels are identical ($l_i = l_j$).
  * $N$: Negative pairs where labels differ ($l_i \neq l_j$).
  * $m$: Margin parameter (default $0.5$).

---

## 📊 Experimental Results (HDFS LogHub Dataset)

The model was pretrained and fine-tuned on the HDFS LogHub dataset using a 3-layer custom Transformer Encoder on an **NVIDIA GeForce RTX 4070 Laptop GPU**.

### 1. Stage 1: Self-Supervised MLM Pretraining (5 Epochs)
* **Validation Token Accuracy**: **97.15%**
* **Validation Perplexity**: **1.08**

### 2. Stage 2: Supervised Fine-Tuning (10 Epochs)
* **Failure Prediction (Anomaly Classification)**:
  * **Accuracy**: **97.60%**
  * **Precision**: **100.00%** (Zero False Positives)
  * **Recall**: **53.97%**
  * **F1-Score**: **70.10%**
  * **ROC-AUC**: **77.28%**
* **Root Cause Analysis (6-Class Failure Categorization)**:
  * **Accuracy**: **97.60%**
  * **Macro F1-Score**: **72.61%**

---

## 🔍 Observing Model Convergence

The training and evaluation scripts output visualization assets under the `plots/` folder:

1. **Pretraining Loss Curves (`plots/pretraining_curves.png`)**: Shows loss decay and validation token accuracy progression.
2. **Fine-Tuning Performance Curves (`plots/finetuning_curves.png`)**: Charts anomaly detection and RCA Macro F1 scores.
3. **t-SNE Embeddings (`plots/test_embeddings_tsne.png`)**: Displays how contrastive embeddings cluster log sequences by their Root Cause Analysis categories, clearly separating Normal events from Write Failures, Timeouts, and Serving errors.
4. **Attention Heatmap (`plots/sample_attention_heatmap.png`)**: Shows visual attention matrices over specific failure events, illustrating how the self-attention heads focus on exception patterns like `java.io.IOException` and `Could not read from stream`.

---

## 🖥️ Streamlit Interactive UI Dashboard

LogMind includes a complete dashboard to interact with the model locally.

### Features:
* **Analytics Panel**: View training history, convergence curves, and t-SNE embedding clusters.
* **Interactive Inference**: Paste raw log sequences or select from preset failure modes to obtain real-time anomaly probabilities, RCA categories, and interactive attention heatmaps.
* **Vector-Similarity Search**: Query a log sequence against the database of indexed log incidents using vector cosine similarity to find matching historic occurrences.
* **Autoregressive Generation**: Select a seed log template prompt and generate subsequent log sequence paths using the causal attention mask.

To launch the dashboard:
```bash
streamlit run app.py
```

---

## 🚀 Execution Instructions

### 1. Requirements & Setup
Configure your virtual environment and install dependencies:
```bash
# Initialize venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch pandas matplotlib seaborn pyyaml streamlit scikit-learn pytest
```

### 2. Run All Unit Tests
Verify the mathematical correctness of attention weights, layer norm equivalence, MLM masking, and CLM shifting:
```bash
python -m pytest
```

### 3. Run Training & Evaluation Pipeline
```bash
python main.py
```
This script executes the self-supervised pretraining stage, transfers encoder weights, runs multi-task fine-tuning, prints final test metrics, and exports all plots.
