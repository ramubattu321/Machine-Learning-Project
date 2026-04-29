# Applied AI & ML Research: NLP, LLMs, Deep Learning & Production Optimization

> 🏆 Presented at the **California Central Valley Research Symposium (2026)**, California State University, Fresno

---

## Overview

An end-to-end AI research system integrating Machine Learning, Deep Learning, Natural Language Processing, and Large Language Models to optimize production processes and enable intelligent decision-making. The system combines structured data modeling with unstructured text understanding, neural architectures, and retrieval-based reasoning to deliver context-aware insights for real-world industrial applications.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Module 1 — Machine Learning for Production Optimization](#module-1--machine-learning-for-production-optimization)
3. [Module 2 — Tokenization, Vector Embedding & Vector Database](#module-2--tokenization-vector-embedding--vector-database)
4. [Module 3 — NLP: Word Embeddings & Language Model Evaluation](#module-3--nlp-word-embeddings--language-model-evaluation)
5. [Module 4 — Fine-Tuning & RAG for LLM Accuracy](#module-4--fine-tuning--rag-for-llm-accuracy)
6. [Module 5 — ML & Deep Learning: Neural Network Architecture Study](#module-5--ml--deep-learning-neural-network-architecture-study)
7. [Module 6 — SLM vs. LLM: Architecture & Deployment Tradeoff Analysis](#module-6--slm-vs-llm-architecture--deployment-tradeoff-analysis)
8. [Results & Key Insights](#results--key-insights)
9. [Tech Stack](#tech-stack)
10. [Project Structure](#project-structure)
11. [How to Run](#how-to-run)

---

## System Architecture

```
Data Sources
       ↓
 ┌─────────────────────────────────────┐
 │  Structured Data  │ Unstructured Data│
 └────────┬──────────┴────────┬─────────┘
          ↓                   ↓
    ML Models           NLP Pipeline
  (Prediction)     (Tokenization + Embeddings)
          ↓                   ↓
          └────────┬──────────┘
                   ↓
        Vector Database (FAISS / ANN)
                   ↓
        RAG System (LLM + Retrieval)
                   ↓
     Final Output: Insights + Recommendations
```

---

## Module 1 — Machine Learning for Production Optimization

 **Notebooks:** `data_preprocessing.ipynb` · `Linear Regression.ipynb` · `multi Regression.ipynb` · `classification_analysis.ipynb` · `random_forest_model.ipynb` · `svm_model.ipynb` · `decision_tree_model.ipynb`
 **Research Poster:** `machine learning_research_poster.pptx.pdf`
 **Data:** `production_data.csv`

### Objective
Predict production performance and identify key performance drivers using supervised machine learning models trained on structured production data.

### Dataset Features
| Feature | Description |
|---------|-------------|
| Temperature | Operating temperature of the machine |
| Pressure | Applied process pressure |
| Humidity | Environmental humidity level |
| Machine Hours | Hours of machine operation |
| Defect Rate | Rate of defective output units |
| Production Output | Total units produced |

### Methodology

**Step 1 — Data Preprocessing** (`data_preprocessing.ipynb`)
- Missing value handling and outlier detection
- Feature normalization and transformation
- Correlation analysis and feature selection
- Train/test split with cross-validation setup

**Step 2 — Model Implementation & Comparison**

| Notebook | Model | Type |
|----------|-------|------|
| `Linear Regression.ipynb` | Linear Regression | Baseline — interpretable |
| `multi Regression.ipynb` | Multiple Regression | Multi-feature linear modeling |
| `svm_model.ipynb` | Support Vector Machine | Kernel-based non-linear |
| `random_forest_model.ipynb` | Random Forest | Ensemble — robust to overfitting |
| `classification_analysis.ipynb` | Classification Models | Categorical output prediction |
| `decision_tree_model.ipynb` | Decision Tree | Interpretable tree-based model |

**Step 3 — Evaluation**
- Metrics: ROC-AUC, F1 Score, Precision, Recall, RMSE
- K-fold cross-validation for generalizability
- Feature importance ranking for production driver identification

### Results
- Ensemble models (Random Forest, Gradient Boosting) significantly outperformed linear baseline
- Feature engineering improved model generalizability across validation folds
- Top production drivers identified: **Machine Hours**, **Temperature**, **Defect Rate**

### Visualization
Correlation heatmap (`correlation.png`) showing feature relationships:

![Correlation Heatmap](correlation.png)

---

## Module 2 — Tokenization, Vector Embedding & Vector Database

 **Reference:** `Tokonization_Vector Embedding_Vector Database (Ramu).pptx.pdf`

### Objective
Build an NLP pipeline that transforms raw text into semantically rich vector representations stored in a searchable vector database — enabling context-aware retrieval beyond keyword matching.

### Pipeline
```
Raw Text
    ↓
Preprocessing (noise removal, normalization)
    ↓
Tokenization (Word / Subword BPE / WordPiece)
    ↓
Vector Embedding (Word2Vec / GloVe / BERT)
    ↓
Vector Database Storage (FAISS / HNSW / Annoy)
    ↓
Query → Embedding → Similarity Search → Results
```

### Tokenization Methods

| Method | Description |
|--------|-------------|
| Word-based | Simple whitespace/punctuation splitting |
| Subword BPE | Byte Pair Encoding — handles rare and unknown words |
| WordPiece | Used by BERT — balances vocabulary size and coverage |

### Embedding Techniques

| Model | Type | Key Strength |
|-------|------|--------------|
| Word2Vec (Skip-gram / CBOW) | Static | Captures local word context |
| GloVe | Static | Global co-occurrence statistics |
| BERT | Contextual | Bidirectional — context-aware representation |

### Vector Search Methods

| Tool | Algorithm | Best For |
|------|-----------|----------|
| FAISS | ANN (GPU-accelerated) | Large-scale fast retrieval |
| HNSW | Graph-based ANN | High accuracy at scale |
| Annoy | Tree-based ANN | Memory-efficient search |

### Results
- Semantic vector search outperformed keyword-based retrieval on contextual relevance
- BERT embeddings captured the strongest semantic relationships
- ANN indexing enabled fast, scalable retrieval over large text corpora

---

## Module 3 — NLP: Word Embeddings & Language Model Evaluation

 **Reference:** `Natural Language Processing(NLP).pdf`

### Objective
Systematically evaluate word representation methods and language model architectures — benchmarking semantic quality and perplexity to identify the strongest approaches for capturing linguistic meaning.

### Word Representation Methods Evaluated
- One-hot encoding and co-occurrence matrices with LSA/SVD dimensionality reduction
- Skip-gram and CBOW Word2Vec models
- GloVe (Global Vectors for Word Representation)

### Semantic Similarity Benchmarks

| Dataset | Description |
|---------|-------------|
| WS353 | WordSim-353 — human-rated word similarity pairs |
| MC | Miller & Charles similarity judgments |
| RG | Rubenstein & Goodenough word pairs |
| SCWS | Stanford Contextual Word Similarity |
| RW | Stanford Rare Words |

>  **GloVe achieved top performance on all five benchmarks**

### Neural Dependency Parser
```
Input Layer → Hidden Layer (ReLU) → Output Layer (Softmax + Cross-Entropy Loss)
```

| Parser | Speed | Notes |
|--------|-------|-------|
| MaltParser (baseline) | ~200 sent/sec | Rule-based |
| MSTParser (baseline) | ~100 sent/sec | Graph-based |
| **Neural Parser (ours)** | **654 sent/sec** | Competitive UAS/LAS accuracy |

### Language Model Perplexity

| Model | Perplexity | Notes |
|-------|------------|-------|
| N-gram | Highest | No long-range context |
| RNN | Medium | Short memory window |
| **2-layer LSTM** | **30 (best)** | Strong long-range dependency modeling |

> 2-layer LSTM outperformed all baselines — confirming transformer-based attention as the strongest architecture for sequence modeling.

---

## Module 4 — Fine-Tuning & RAG for LLM Accuracy

 **Reference:** `Fine-Tuning & RAG.pdf`

### Objective
Reduce LLM hallucination anomaly by implementing and comparing iterative retrieval strategies and adaptive fine-tuning across multiple LLM architectures.

### Techniques Implemented

**Iter-RetGen — Iterative RAG**
```
Query → LLM Generation → Retrieval → Re-generation → ... → Final Answer
```
- Cross-validates generated responses against external knowledge sources
- Iteratively refines output accuracy through repeated retrieval passes

**ACTD — Answer Candidates and Task Decomposition**
```
Query
  ↓
LLM confidence check
  ↓
Sufficient knowledge?  →  Yes → Direct answer
       ↓ No
Task decomposition into sub-questions
       ↓
Targeted retrieval per sub-question
       ↓
Composed final answer
```
- Adaptive — only retrieves when LLM internal knowledge is insufficient
- Reduces unnecessary retrieval overhead on simpler queries

### Benchmark Results (across 4 LLM architectures)

| Strategy | Accuracy | Efficiency |
|----------|----------|------------|
| No retrieval (baseline) | Lowest | Highest |
| Single-pass RAG | Moderate | High |
| Iter-RetGen | High | Moderate |
| **ACTD** | **Matched / exceeded Iter-RetGen** | **Best** |

> ACTD demonstrated that structured question decomposition produces more accurate and contextually grounded responses than iterative retrieval alone — with lower computational cost.

---

## Module 5 — ML & Deep Learning: Neural Network Architecture Study

 **Reference:** `Machine Learning and Deep Learning.pdf`

### Objective
Analyze and compare machine learning and deep learning architectures — understanding how neural networks process information and when deep learning outperforms classical ML.

### Neural Network Architecture (Digit Recognition)
```
Input Layer:    784 nodes  (28×28 pixel flattened image)
       ↓
Hidden Layer 1: Edge detection          (low-level features)
       ↓
Hidden Layer 2: Shape detection         (mid-level features)
       ↓
Hidden Layer 3: Pattern composition     (high-level features)
       ↓
Output Layer:   10 nodes   (digits 0–9, Softmax activation)
```

**Key Concepts Evaluated**
- Activation functions: ReLU, Sigmoid, Softmax — role and impact on learning
- Backpropagation: gradient computation and weight update mechanics
- Regularization: dropout and batch normalization for generalization

### ML vs. Deep Learning Comparison

| Dimension | Machine Learning | Deep Learning |
|-----------|-----------------|---------------|
| Data requirements | Small–medium datasets | Large datasets required |
| Feature engineering | Manual extraction needed | Automatic feature learning |
| Computational cost | Low–Medium | High (GPU recommended) |
| Interpretability | Higher | Lower (black box) |
| Best use case | Tabular / structured data | Images, text, audio, video |

### Emerging Techniques Studied

| Technique | Description |
|-----------|-------------|
| Transfer Learning | Reuse pretrained model weights for new tasks |
| Explainable AI (XAI) | Interpret and explain black-box model predictions |
| Federated Learning | Privacy-preserving distributed model training |

---

## Module 6 — SLM vs. LLM: Architecture, Scaling & Deployment Tradeoff Analysis

 **Reference:** `SLM and LLM.pdf`

### Objective
Provide a rigorous technical comparison of Small Language Models (SLMs) and Large Language Models (LLMs) to inform practical deployment decisions for industrial AI systems.

### Scale Definitions

| Category | Parameter Range | Examples |
|----------|----------------|----------|
| SLM | 100M – 5B | DistilBERT, Phi-2, Mistral 7B |
| LLM | 5B+ | GPT-4, LLaMA 70B, Claude |

### Five-Dimension Analysis

**1. Transformer Foundations** — Both share the same core architecture but differ in depth, width, attention heads, and training data scale.

**2. Attention Mechanisms**
- SLMs: Efficient attention variants (grouped-query, sliding window) for reduced memory
- LLMs: Full multi-head attention supporting broader context and richer reasoning

**3. Scaling Laws** — Based on Kaplan et al. and Hoffmann et al. (Chinchilla): optimal model size depends on compute budget and dataset size. Diminishing returns observed beyond certain thresholds.

**4. Efficiency Tradeoffs**

| Factor | SLM | LLM |
|--------|-----|-----|
| Latency | ✅ Low | ❌ High |
| Memory footprint | ✅ Low | ❌ High |
| Cost per query | ✅ Low | ❌ High |
| Edge / on-device | ✅ Yes | ❌ No |
| Open-ended reasoning | ❌ Limited | ✅ Strong |
| Complex multi-step tasks | ❌ Limited | ✅ Strong |

**5. Recommended Hybrid Architecture**
```
Incoming Query
       ↓
  Complexity Router
       ↓
Simple / Domain Query  →  SLM  (fast, local inference)
Complex / Open Query   →  LLM  (orchestration + reasoning)
       ↓
    Response
```

> Hybrid architectures deliver the best of both worlds — LLMs for complex reasoning, SLMs for speed, cost, and privacy.

---

## Results & Key Insights

| Finding | Detail |
|---------|--------|
| Best ML model | Random Forest & Gradient Boosting outperformed all single models |
| Best word embedding | GloVe — top scores across all 5 NLP benchmarks |
| Best language model | 2-layer LSTM — perplexity of 30, best among N-gram, RNN, LSTM |
| Best RAG strategy | ACTD matched or exceeded Iter-RetGen across all 4 LLM architectures |
| Parser speed | Neural dependency parser — 654 sentences/second |
| Semantic retrieval | Vector search outperformed keyword matching on contextual relevance |
| SLM vs. LLM | Hybrid architecture recommended for industrial deployment |

### Key Insights
- Data quality and feature engineering strongly impact ML model performance
- Embedding-based retrieval improves information relevance over keyword matching
- RAG reduces LLM hallucination — ACTD does so more efficiently than iterative retrieval
- Hybrid AI systems (ML + NLP + LLM) outperform standalone models in complex environments
- SLMs and LLMs are complementary — hybrid deployment maximizes speed and reasoning quality

---

## Tech Stack

```
Language:       Python
ML:             Scikit-learn, Random Forest, Gradient Boosting, SVM, Regression, Decision Tree
Deep Learning:  PyTorch, TensorFlow, Neural Networks
NLP:            NLTK, Word2Vec, GloVe, BERT, Transformers, Hugging Face
LLM:            RAG, ACTD, LLM Fine-Tuning, SLM/LLM Architecture Analysis
Vector Search:  FAISS, HNSW, Annoy, ANN
Data:           Pandas, NumPy, SciPy
Visualization:  Matplotlib, Seaborn
```

---

## Project Structure

```
├── production_data.csv                              # Structured production dataset
│
├── data_preprocessing.ipynb                        # Data cleaning, EDA, feature engineering
├── Linear Regression.ipynb                         # Simple linear regression model
├── multi Regression.ipynb                          # Multiple regression model
├── classification_analysis.ipynb                   # Classification model analysis
├── random_forest_model.ipynb                       # Random Forest model
├── svm_model.ipynb                                 # Support Vector Machine model
├── decision_tree_model.ipynb                       # Decision Tree model
│
├── Tokonization_Vector Embedding_Vector             # NLP pipeline — tokenization,
│   Database (Ramu).pptx.pdf                        #   embeddings & vector DB
├── Natural Language Processing(NLP).pdf            # Word embeddings & LM evaluation
├── Fine-Tuning & RAG.pdf                           # RAG & LLM fine-tuning research
├── Machine Learning and Deep Learning.pdf          # Neural network architecture study
├── SLM and LLM.pdf                                 # SLM vs LLM tradeoff analysis
├── machine learning_research_poster.pptx.pdf       # Research poster (Symposium 2026)
│
├── visualization.py                                # Visualization script
├── correlation.png                                 # Correlation heatmap output
├── requirements.txt                                # Python dependencies
└── README.md                                       # Project documentation
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/ramubattu321/machine-learning-production-optimization.git
cd machine-learning-production-optimization

# Install dependencies
pip install -r requirements.txt

# Run visualization
python visualization.py

# Open any notebook
jupyter notebook data_preprocessing.ipynb
jupyter notebook random_forest_model.ipynb
jupyter notebook svm_model.ipynb
```

---

## Research Contribution

This project demonstrates how integrating machine learning, deep learning, NLP, and LLM-based systems can create scalable, intelligent solutions for real-world industrial problems. It highlights the importance of hybrid AI architectures for combining predictive modeling, semantic understanding, and contextual reasoning.

Presented at the **California Central Valley Research Symposium (2026)**, California State University, Fresno.

---

## Author

**Ramu Battu**
MS in Data Analytics — California State University, Fresno
*Non-Resident Tuition Waiver (NRTW) Scholarship Recipient*
