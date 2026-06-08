# AI Data Scientist / AI Engineer Learning Roadmap

This roadmap is designed for a quant analyst who already has strong foundations in statistics, structured data, risk modeling, and partial data engineering. The goal is not to relearn everything from scratch. The goal is to convert existing quantitative strength into production-grade AI capability: modern ML engineering, LLM systems, model risk governance, and AI agent engineering.

## Starting Point and Strategy

**Assumption:** Phase 0, Phase 1, and part of Phase 2 are already familiar from quant/risk work. Treat those sections as quick refreshes and validation checkpoints, not long study blocks.

**Primary learning principle:** every phase should produce an artifact that can be shown, reviewed, tested, or deployed. Reading and courses are useful, but the main proof of progress is a working project, an evaluation report, or a production-style system.

**Suggested weekly rhythm:**
* **Theory:** 25% of time, focused on math and architecture.
* **Implementation:** 45% of time, building notebooks, scripts, APIs, and pipelines.
* **Evaluation and audit:** 20% of time, writing metrics, tests, drift checks, and validation reports.
* **Review and documentation:** 10% of time, turning work into reusable notes and portfolio artifacts.

---

## Phase 0: Core Working Stack Refresh
**Status:** Mostly complete from quant analyst experience. Use this as a checklist.

### Core Skills
* **Python:** NumPy, pandas, scikit-learn, typing, virtual environments, packaging basics.
* **SQL:** joins, window functions, aggregations, CTEs, time-series queries, feature extraction.
* **Git and Shell:** branching, diffs, commits, command-line navigation, reproducible scripts.
* **Experiment Hygiene:** notebook structure, random seeds, train/validation/test separation, config files.

### Validation Deliverable
* Create a small reproducible ML project template with `data/`, `notebooks/`, `src/`, `tests/`, and `README.md`.
* Include one CLI command or notebook that trains a baseline model from raw data.

### Success Criteria
* You can rerun the experiment from a clean environment.
* Another engineer can understand the project structure without asking you.

---

## Phase 1: Classical ML and Risk Audit Foundations
**Status:** Mostly complete. Treat this as a fast consolidation phase.

**Focus:** Structured data modeling, feature safety, and the core algorithms behind risk, fraud detection, credit scoring, and portfolio analytics.

### 1. Classical Statistics and Linear Models
* **Linear and Logistic Regression:** Loss functions such as Mean Squared Error and Binary Cross-Entropy; optimization with Gradient Descent variants.
* **Regularization:** Mathematical mechanics of Lasso ($L_1$) and Ridge ($L_2$); how penalties reduce overfitting and enforce feature sparsity.
* **Model Diagnostics:** Bias-variance tradeoff, learning curves, residual analysis, multicollinearity, calibration, and stability.

### 2. Industrial Risk Models
* **Decision Trees:** Entropy, Information Gain, Gini Impurity, pruning, and interpretability tradeoffs.
* **Ensemble Methods:** Bagging versus boosting, Random Forest, Gradient Boosting Decision Trees.
* **XGBoost and LightGBM:** Second-order Taylor approximation, histogram algorithms, leaf-wise growth, monotonic constraints, missing-value handling.

### 3. Data and Model Audit Standards
* **Feature Engineering Traps:** Target leakage, temporal leakage, survivorship bias, proxy variables, and stale features.
* **Imbalanced Data Auditing:** Class weights, threshold tuning, under-sampling, SMOTE, and when synthetic sampling is dangerous.
* **Robustness Evaluation:** PSI, KS statistic, calibration curves, ROC-AUC versus PR-AUC, lift charts, and slice-level performance.

### Validation Deliverable
* Build or refactor one structured-risk model using LightGBM or XGBoost.
* Add leakage checks, class imbalance strategy, calibration analysis, feature importance, PSI drift report, and a short model validation memo.

### Success Criteria
* You can explain why the chosen metric is appropriate.
* You can identify which features are risky from a leakage, fairness, or drift perspective.
* You can present the model as both a data scientist and a model risk reviewer.

---

## Phase 2: Data Engineering for ML
**Status:** Partially complete. Prioritize gaps that are less common in quant analyst workflows.

**Focus:** Turning raw data into reliable, testable, production-ready ML features.

### 1. Data Pipeline Foundations
* **Batch Pipelines:** Ingest, clean, transform, validate, and materialize datasets.
* **Data Modeling:** Star schema basics, fact/dimension tables, time-aware joins, slowly changing dimensions.
* **Feature Pipelines:** Point-in-time correctness, training-serving skew, feature freshness, and feature reuse.

### 2. Data Quality and Validation
* **Data Contracts:** Expected columns, types, nullability, value ranges, uniqueness, and referential integrity.
* **Testing Tools:** Great Expectations, Pandera, dbt tests, or lightweight custom checks.
* **Monitoring:** Missingness, distribution shift, schema drift, delayed feeds, outlier explosions.

### 3. Orchestration and Storage
* **Workflow Orchestration:** Airflow, Dagster, Prefect, or cron-based MVP pipelines.
* **Storage Patterns:** Parquet, partitioning, warehouse tables, object storage, and local reproducible data snapshots.
* **Lineage:** Tracking data sources, transformations, model inputs, and model outputs.

### Validation Deliverable
* Build an end-to-end feature pipeline for a risk or finance dataset.
* Include point-in-time joins, data validation checks, feature documentation, and a repeatable training dataset export.

### Success Criteria
* You can prove the training data does not use future information.
* You can rerun the pipeline for a new date range.
* Failed data quality checks stop the pipeline or produce clear warnings.

---

## Phase 3: Deep Learning and Computational Frameworks
**Focus:** Moving from human-engineered features to learned representations, while gaining enough framework knowledge to inspect and debug neural networks.

### 1. Deep Learning Theory
* **Forward Propagation and MLPs:** Matrix multiplication, activations, representation learning.
* **Backward Propagation:** Chain Rule, automatic differentiation, computational graphs.
* **Neural Network Pathology:** Vanishing gradients, exploding gradients, residual connections, LayerNorm versus BatchNorm, Xavier and He initialization.
* **Optimizer Evolution:** SGD $\rightarrow$ Momentum $\rightarrow$ Adam $\rightarrow$ AdamW.

### 2. PyTorch Framework Mastery
* **Tensor Operations:** `view`, `reshape`, `permute`, `squeeze`, broadcasting rules, and device placement.
* **Training Loops:** Dataset, DataLoader, model, loss, optimizer, scheduler, checkpointing.
* **Hooks and Interpretability:** Forward hooks, backward hooks, activation extraction, gradient inspection.
* **Hardware Awareness:** GPU memory, batch size tradeoffs, mixed precision, profiling basics.

### Validation Deliverable
* Implement a PyTorch neural network training loop from scratch.
* Add checkpointing, metrics, TensorBoard or equivalent logging, and at least one forward hook for activation inspection.

### Success Criteria
* You can debug tensor shape errors quickly.
* You can explain what happens during backpropagation.
* You can inspect intermediate activations and gradients without treating the model as a black box.

---

## Phase 4: Large Language Models and Fine-Tuning
**Focus:** Understanding Transformer internals, using LLMs effectively, and quantifying generative risk.

### 1. Transformer Architecture
* **Attention Mechanisms:** Scaled Dot-Product Attention, Multi-Head Attention, $Q$, $K$, and $V$ projections.
* **Scaling Factor:** Why attention divides by $\sqrt{d_k}$, from variance and gradient stability perspectives.
* **Positional Encodings:** Sinusoidal encodings, learned positions, RoPE.
* **Inference Bottlenecks:** KV cache, auto-regressive decoding, memory-bound inference, latency and throughput.

### 2. Embeddings, RAG, and Retrieval
* **Embeddings:** Similarity search, embedding model selection, chunking strategies, metadata filtering.
* **Retrieval:** BM25, dense search, hybrid sparse-dense search, reranking, query rewriting.
* **RAG Failure Modes:** Missing context, irrelevant context, stale context, conflicting sources, hallucinated citations.

### 3. PEFT and Quantization
* **LoRA and QLoRA:** Low-rank adaptation, $W_0 + \Delta W = W_0 + BA$, rank choice, target modules.
* **Quantization:** 4-bit and 8-bit quantization, NF4, latency-memory-quality tradeoffs.
* **Distributed Training Concepts:** DDP, Tensor Parallelism, Pipeline Parallelism, and ZeRO memory stages.

### 4. Alignment and Generative Risk Metrics
* **Alignment Techniques:** SFT, RLHF, DPO, preference data, reward modeling basics.
* **Semantic Evaluation:** BERTScore, embedding cosine similarity, answer relevance.
* **Risk Auditing Metrics:** Hallucination rate, groundedness, toxicity, refusal rates, jailbreak resistance, catastrophic forgetting.

### Validation Deliverable
* Build a finance or model-risk RAG assistant over a controlled document set.
* Add a golden evaluation dataset, retrieval metrics, hallucination checks, citation checks, latency tracking, and cost tracking.

### Success Criteria
* The system can distinguish "not found in source" from a confident answer.
* You can measure retrieval quality separately from generation quality.
* You can explain the tradeoff between chunking, reranking, latency, and answer quality.

---

## Phase 5: MLOps and Production AI Systems
**Focus:** Taking models and LLM applications from notebooks to repeatable services.

### 1. Experiment and Model Lifecycle
* **Experiment Tracking:** MLflow, Weights & Biases, or lightweight local tracking.
* **Model Registry:** Versioning models, metrics, parameters, datasets, and approval status.
* **Reproducibility:** Environment locking, deterministic seeds, artifact storage, data versioning.

### 2. Serving and Deployment
* **Inference APIs:** FastAPI, request validation, batch inference, online inference.
* **Containers:** Dockerfiles, dependency isolation, image size, secrets handling.
* **Deployment Basics:** Cloud VM, serverless endpoint, managed ML platform, or internal service.

### 3. Monitoring and Operations
* **Model Monitoring:** Data drift, prediction drift, performance decay, latency, error rate.
* **Operational Resilience:** Retries, timeouts, circuit breakers, fallback behavior.
* **Governance:** Approval gates, audit logs, model cards, data cards, validation reports.

### Validation Deliverable
* Deploy one ML model or LLM/RAG application as a local or cloud API.
* Include experiment tracking, model versioning, Docker, simple CI checks, monitoring logs, and a model card.

### Success Criteria
* You can reproduce the deployed artifact from code and configuration.
* You can roll back to a previous model version.
* You can explain how the system fails and how it is monitored.

---

## Phase 6: AI Agent Engineering and Multi-Agent Systems
**Focus:** Treating the LLM as one component inside a deterministic, stateful software system.

### 1. Single-Agent Architecture Patterns
* **Reasoning Patterns:** ReAct, Plan-and-Solve, planner-executor, reflection loops.
* **Tool Calling:** JSON Schema, function calling, typed inputs, validation, retries.
* **Memory Architectures:** Sliding windows, summarization, vector memory, advanced RAG, reranking.

### 2. Multi-Agent Topologies and Orchestration
* **State-Machine Architectures:** LangGraph-style nodes, edges, conditional routing, durable state.
* **Collaboration Patterns:** Orchestrator-workers, chains, supervisor-evaluator, reviewer-reviser.
* **Audit Agent Pattern:** A specialized agent validates another agent's output with code checks, semantic checks, source checks, or policy checks.

### 3. Production-Grade Agent Engineering
* **Sandbox Execution:** Isolated tool execution, Docker, restricted file access, SQL safety, prompt injection resistance.
* **Robustness:** Invalid JSON repair, schema validation, tool timeout handling, rate-limit handling.
* **Agent Evaluation:** Ragas, TruLens, custom golden sets, trace review, task success rate, groundedness.

### Validation Deliverable
* Build a model-risk audit agent that can inspect a model report, run structured checks, call tools, produce findings, and cite evidence.
* Add an evaluator agent or deterministic evaluator that checks the final output before release.

### Success Criteria
* The agent does not proceed when required evidence is missing.
* Tool calls are typed, validated, logged, and recoverable.
* The output is auditable enough for a human reviewer.

---

## Capstone: Finance AI Risk Intelligence System
**Focus:** Combine the full stack into one coherent portfolio project.

### Project Idea
Build an AI system that supports model risk or investment risk analysis. It should combine:
* A structured ML model, such as default prediction, anomaly detection, fraud detection, or portfolio risk classification.
* A data pipeline with point-in-time feature generation and data quality checks.
* A RAG layer over model documentation, policy documents, research notes, or market reports.
* An agent workflow that can answer questions, run validation checks, and produce an audit-style summary.
* A deployed API or app with logging, evaluation, and monitoring.

### Required Artifacts
* Source code with reproducible setup.
* Training dataset generation pipeline.
* Model training and evaluation report.
* RAG evaluation report.
* Model card and data card.
* Agent evaluation report.
* Deployment instructions.

### Success Criteria
* A reviewer can run the project end to end.
* The system has measurable quality gates.
* The project demonstrates quant modeling, AI engineering, data engineering, LLM evaluation, and model risk thinking in one place.

---

## Suggested Order for Your Background

Since you already have Phase 0, Phase 1, and part of Phase 2:

1. Spend **1 week** turning Phase 1 knowledge into a polished risk-model validation artifact.
2. Spend **2-3 weeks** closing Phase 2 data engineering gaps with point-in-time feature pipelines and validation checks.
3. Spend **3-4 weeks** on Phase 3 PyTorch implementation and debugging depth.
4. Spend **4-6 weeks** on Phase 4 LLM/RAG systems and evaluation.
5. Spend **3-4 weeks** on Phase 5 MLOps and deployment.
6. Spend **4-6 weeks** on Phase 6 agent systems.
7. Spend **4-8 weeks** on the capstone.

The fastest route for you is not more theory first. It is to convert your quant foundation into production artifacts, then layer LLM engineering and agent evaluation on top.
