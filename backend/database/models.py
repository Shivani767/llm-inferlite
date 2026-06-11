from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON, Boolean, Enum
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import enum

Base = declarative_base()

class ModelFormat(enum.Enum):
    HF = "huggingface"
    ONNX = "onnx"
    GGUF = "gguf"
    TRT = "tensorrt"

class Model(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True)
    family = Column(String) # Llama, Mistral, etc.
    architecture = Column(String)
    parameters = Column(String) # e.g., "8B"
    context_length = Column(Integer)
    tokenizer_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    versions = relationship("ModelVersion", back_populates="model", cascade="all, delete-orphan")

class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("models.id"))
    version_tag = Column(String) # v1.0, fp16, etc.
    format = Column(Enum(ModelFormat))
    sha256 = Column(String)
    size_bytes = Column(Integer)
    metadata_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    model = relationship("Model", back_populates="versions")
    benchmarks = relationship("Benchmark", back_populates="model_version")
    quantizations = relationship("Quantization", back_populates="model_version")

class Benchmark(Base):
    __tablename__ = "benchmarks"
    
    id = Column(Integer, primary_key=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"))
    runtime = Column(String) # vllm, ort, etc.
    batch_size = Column(Integer)
    avg_latency_ms = Column(Float)
    p99_latency_ms = Column(Float)
    throughput_tps = Column(Float)
    ttft_ms = Column(Float)
    memory_usage_gb = Column(Float)
    config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    model_version = relationship("ModelVersion", back_populates="benchmarks")

class Quantization(Base):
    __tablename__ = "quantizations"
    
    id = Column(Integer, primary_key=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"))
    method = Column(String) # GPTQ, AWQ, etc.
    precision = Column(String) # INT4, INT8
    mse_error = Column(Float)
    perplexity_delta = Column(Float)
    memory_reduction_pct = Column(Float)
    config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    model_version = relationship("ModelVersion", back_populates="quantizations")

class Evaluation(Base):
    __tablename__ = "evaluations"
    
    id = Column(Integer, primary_key=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"))
    benchmark_name = Column(String) # MMLU, GSM8K
    score = Column(String)
    delta_from_base = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Experiment(Base):
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    cuda_version = Column(String)
    driver_version = Column(String)
    hardware = Column(String)
    config = Column(JSON)
    results = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
