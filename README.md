# LLM-A-to-Z  
> This repository is expected to cover every concept, technique, and practice required to confidently say:  
> **"I’m a master of Large Language Models (LLMs)"**  
>  
> The checklist below represents the topics covered.  
> As this repo grows, the list will be updated to mark what’s been covered ✅

---

## 1. Core Mathematical Foundations
> This section is cover in https://github.com/PandukaBandara99/ML-Books Repo
- [✅] Linear Algebra (matrix operations, eigenvalues, orthogonality)  
- [✅] Probability & Statistics (Bayes theorem, entropy, information theory)  
- [✅] Calculus (partial derivatives, chain rule, gradient computation)  
- [✅] Optimization Algorithms (SGD, Adam, RMSProp, AdaFactor)  
- [✅] Backpropagation & Automatic Differentiation  
- [✅] Loss Functions (Cross-Entropy, KL Divergence, Contrastive Loss)  
- [✅] Initialization & Regularization ( LayerNorm, Dropout)  



##  2. NLP Fundamentals

- [✅] Tokenization (BPE, WordPiece, SentencePiece)  
- [✅] Vocabulary construction & subword merging  
- [✅] Word embeddings (Word2Vec)  
- [✅] Contextual embeddings (ELMo, BERT)  
- [✅] Sequence modeling (RNNs, LSTMs, GRUs)  
- [✅] Attention mechanism — motivation & formulation  
- [✅] Transformer evolution  

---

## ⚙️ 3. Transformer Architecture Mastery

- [✅] Self-attention mechanism (scaled dot-product, query/key/value math)  
- [✅] Multi-head attention — intuition and tensor shapes  
- [✅] Feedforward networks and non-linear activations (GELU, ReLU)  
- [✅] Positional encodings
- [✅] Residual connections and LayerNorm internals  
- [✅] Encoder, Decoder, and Decoder-only variants (BERT, T5, GPT)  
- [✅] Transformer from scratch (PyTorch / TensorFlow)  
- [✅] Parameter initialization, gradient flow, and stability analysis  
- [✅] Handling variable sequence lengths and masking  
- [✅] Attention efficiency improvements (FlashAttention, Linformer, Performer)  


## 4. Pretraining & Training Pipelines

- [✅] Pretraining objectives (Causal LM, Masked LM, Next Sentence Prediction)  
- [✅] Dataset curation, cleaning, and tokenization  
- [✅] DataLoader and batching strategies for large-scale corpora  
- [✅] Optimizer scheduling (Warmup, Cosine decay, One-cycle)  
- [✅] Mixed precision training (FP16, BF16)  
- [✅] Gradient checkpointing and accumulation  
- [✅] Distributed training (DataParallel, FSDP, ZeRO, DeepSpeed)  
- [✅] Evaluation metrics: Perplexity, BLEU, ROUGE, Accuracy  
- [✅] Logging and experiment tracking (MLflow)  
- [✅] Checkpointing and model saving best practices  



## 5. Fine-tuning, Alignment & Adaptation

- [ ] Supervised fine-tuning (SFT) workflows  
- [ ] LoRA / QLoRA / Prefix / Adapter tuning  
- [ ] Parameter-efficient fine-tuning (PEFT)  
- [ ] RLHF — Reinforcement Learning from Human Feedback  
- [ ] Reward model training & preference datasets  
- [ ] Direct Preference Optimization (DPO)  
- [ ] Reinforcement Learning with AI Feedback (RLAIF)  
- [ ] Instruction-tuning & multi-turn dialogue adaptation  
- [ ] Safety alignment and refusal mechanisms  
- [ ] Evaluation on alignment benchmarks (TruthfulQA, MMLU)  


## 6. Scaling Laws & Efficiency Techniques

- [ ] Understanding scaling laws (model size, data, compute)  
- [ ] Activation checkpointing and gradient sharding  
- [ ] Memory-efficient attention (FlashAttention, xFormers)  
- [ ] Quantization (8-bit, 4-bit, GPTQ, AWQ)  
- [ ] Pruning and knowledge distillation  
- [ ] Efficient optimizers (Adafactor, Lion, Sophia)  
- [ ] Large-batch and distributed gradient strategies  
- [ ] Pipeline parallelism vs. tensor parallelism  
- [ ] Sharded model storage and loading (FSDP, ZeRO)  
- [ ] Profiling and optimization with PyTorch Profiler / NVTX  



## 7. Inference & Decoding Strategies

- [ ] Greedy decoding and beam search  
- [ ] Top-k and top-p (nucleus) sampling  
- [ ] Temperature scaling and diversity penalties  
- [ ] Repetition, presence, and frequency penalties  
- [ ] Caching KV states for fast autoregressive decoding  
- [ ] Streaming inference for chat-based models  
- [ ] Efficient serving with vLLM, TensorRT-LLM, or ONNX  
- [ ] Quantized inference pipelines (INT8, 4-bit, NF4)  
- [ ] Benchmarking latency and throughput metrics  


## 8. Retrieval-Augmented Generation (RAG)

- [ ] Embedding models and vector representations  
- [ ] Vector databases (FAISS, Chroma, Pinecone)  
- [ ] Document chunking and context window management  
- [ ] Context injection and grounding techniques  
- [ ] Hybrid retrieval (dense + sparse)  
- [ ] Caching and reranking strategies  
- [ ] Evaluating retrieval quality (Recall@K, MRR)  
- [ ] Implementing RAG pipelines (LangChain, LlamaIndex)  


## 9. Deployment & Integration

- [ ] Serving LLMs with FastAPI, Flask, or Gradio  
- [ ] REST / gRPC / WebSocket APIs  
- [ ] Frontend chat interface (React, Next.js)  
- [ ] Cloud deployments (AWS SageMaker, GCP Vertex AI, Azure)  
- [ ] Model serialization and versioning (TorchScript, Safetensors)  
- [ ] Load balancing & autoscaling for inference  
- [ ] Monitoring (Prometheus, Grafana)  
- [ ] CI/CD workflows for LLM applications  
- [ ] Containerization (Docker) and orchestration (Kubernetes)  
- [ ] Security, rate limiting, and data privacy in deployment  



## 10. Multimodal & Advanced LLM Topics

- [ ] Vision-Language Models (CLIP, BLIP, Flamingo)  
- [ ] Audio-Language Models (Whisper, AudioLM)  
- [ ] Video + Text models (Sora, Kosmos)  
- [ ] Tool-augmented models (LangChain, OpenAI Function Calling)  
- [ ] Agentic LLMs with planning and memory  
- [ ] ReAct and Reflexion-style architectures  
- [ ] Memory-augmented transformers and retrieval memory  
- [ ] Continual and lifelong learning approaches  
- [ ] Self-correcting and self-improving LLMs  
- [ ] Integration with external APIs, databases, and systems  



## 11. Evaluation, Safety & Ethics

- [ ] Benchmarking with MMLU, GSM8K, ARC, HELM  
- [ ] Factuality, consistency, and reasoning evaluation  
- [ ] Bias, fairness, and toxicity detection  
- [ ] Explainability and interpretability tools (Captum, SHAP)  
- [ ] Red-teaming and adversarial prompt testing  
- [ ] Safety layers and content filtering  
- [ ] Human evaluation and feedback loops  
- [ ] Ethical deployment guidelines and documentation  

## 12. Ecosystem & Framework Mastery

- [ ] Hugging Face Transformers, Datasets, PEFT, Accelerate  
- [ ] PyTorch Lightning / DeepSpeed / Megatron-LM  
- [ ] LangChain / LlamaIndex / Haystack  
- [ ] OpenAI API / Anthropic / Cohere SDKs  
- [ ] SentenceTransformers, FAISS, ChromaDB  
- [ ] vLLM / Ollama / LM Studio for inference  
- [ ] Weights & Biases, MLflow for experiment tracking  
- [ ] Triton / TensorRT / ONNX Runtime  
- [ ] Model evaluation libraries (lm-eval-harness, AlpacaEval)  
- [ ] Git, DVC, and reproducibility best practices  


## 13. Research & Cutting-Edge Directions

- [ ] Sparse and Mixture-of-Experts (MoE) Transformers  
- [ ] State Space Models (S4, Mamba)  
- [ ] Long-context modeling (Attention Sink, Mamba, FlashAttention-2)  
- [ ] Token-free and continuous LLMs  
- [ ] Modular and compositional reasoning models  
- [ ] Self-reflection, CoT (Chain-of-Thought), and ToT (Tree-of-Thought)  
- [ ] Multi-agent LLM systems  
- [ ] Alignment and interpretability research trends  
- [ ] Efficiency benchmarking and LLMOps  

