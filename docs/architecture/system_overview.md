# InferLite: High-Level Architecture

## 1. System Vision
InferLite is an "Inference Operating System" designed for deep systems engineering research into LLM optimization. It bridges the gap between raw hardware capabilities and high-level application requirements by providing an automated, research-grade optimization pipeline.

## 2. Component Diagram

```mermaid
graph TD
    User([Optimization Engineer]) --> API[FastAPI Gateway]
    
    subgraph "Control Plane (Control & Metadata)"
        API --> Registry[Model Registry]
        API --> Advisor[AI Optimization Advisor]
        API --> ExpTracker[Experiment Tracker]
        Registry --> MetaDB[(PostgreSQL)]
    end
    
    subgraph "Data Plane (Inference & Research)"
        API --> TaskQueue[Celery + Redis]
        TaskQueue --> Workers[Research Workers]
        
        subgraph "Optimization Engines"
            Workers --> QuantEngine[Quantization Engine]
            Workers --> ONNXPipeline[ONNX Opt Pipeline]
            Workers --> AutoTune[Auto-Tuning Engine]
        end
        
        subgraph "Research Labs"
            Workers --> KVCacheLab[KV Cache Lab]
            Workers --> SpecDecoding[Speculative Decoding Lab]
        end
        
        subgraph "Runtime Abstraction"
            Workers --> VLLM[vLLM]
            Workers --> TRT[TensorRT-LLM]
            Workers --> LlamaCpp[llama.cpp]
            Workers --> ORT[ONNX Runtime]
        end
    end
    
    subgraph "Observability & Intelligence"
        Workers --> Telemetry[GPU Telemetry Engine]
        Telemetry --> MetricsDB[(ClickHouse)]
        Workers --> Eval[Evaluation Framework]
        API --> Planner[Cost-Aware Deployment Planner]
    end
```

## 3. Core Modules
1.  **Model Registry**: Centralized management of model metadata, versions, and hashes.
2.  **Multi-Runtime Engine**: Unified interface for heterogeneous inference backends.
3.  **Quantization Engine**: Research into weight and activation compression (GPTQ, AWQ, INT8).
4.  **ONNX Optimization**: Graph-level transformations and fusion.
5.  **Auto-Tuning**: Bayesian or Grid search for optimal runtime/quantization/batch-size configs.
6.  **KV Cache Lab**: Advanced memory management (PagedAttention, Prefix Cache).
7.  **GPU Telemetry**: Real-time power, thermal, and utilization tracking.

## 4. Technology Stack
- **Language**: Python 3.12 (Strongly Typed)
- **Framework**: FastAPI (Asynchronous)
- **Database**: PostgreSQL (Metadata), ClickHouse (Telemetry), Redis (Task Queue)
- **Infrastructure**: Kubernetes, Docker, GitHub Actions
