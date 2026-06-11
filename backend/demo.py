import asyncio
import json
from services.research_model_service import ResearchModelService
from database.session import SessionLocal, engine
from database.models import Base
from benchmarking.benchmark_farm import BenchmarkFarm
from advisor.ai_advisor import AIResearchAdvisor
from telemetry.gpu_engine import GPUTelemetryEngine

async def run_demo():
    print("=== InferLite: LLM Inference OS - Research Demo ===\n")
    
    # Initialize DB with unique name for demo
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    demo_engine = create_engine("sqlite:///./demo_research.db")
    DemoSession = sessionmaker(autocommit=False, autoflush=False, bind=demo_engine)
    
    Base.metadata.create_all(bind=demo_engine)
    db = DemoSession()
    
    try:
        # 1. Model Registry Demo
        print("[1] Registering Model...")
        from database.models import Model
        existing_model = db.query(Model).filter(Model.name == "Llama-3-8B-Instruct").first()
        if not existing_model:
            mock_model = Model(name="Llama-3-8B-Instruct", family="Llama", architecture="llama", parameters="8B")
            db.add(mock_model)
            db.commit()
            db.refresh(mock_model)
        else:
            mock_model = existing_model
        
        model_name = mock_model.name
        print(f"Model registered with ID: {mock_model.id}\n")

        # 2. Benchmark Farm Demo
        print("[2] Running Automated Benchmark Farm (Quantize -> Benchmark -> Evaluate)...")
        farm = BenchmarkFarm(db)
        results = await farm.process_new_model(model_id=mock_model.id) 
        
        print(f"Quantization Methods Tested: GPTQ, AWQ")
        for eval_res in results.get("evaluation_results", []):
            method = eval_res["method"]
            metrics = eval_res["metrics"]
            delta = eval_res["delta_analysis"]
            print(f"  > {method.upper()}: Accuracy Retention: {delta['accuracy_retention_pct']:.2f}%, Speedup: {delta['latency_speedup_x']:.2f}x")
        print("")

        # 3. GPU Telemetry Demo
        print("[3] Collecting GPU Telemetry...")
        gpu_engine = GPUTelemetryEngine()
        metrics = gpu_engine.collect_metrics()
        for m in metrics:
            print(f"  GPU {m.gpu_id}: {m.name}")
            print(f"  Utilization: {m.utilization_gpu:.1f}%, Temp: {m.temperature_c:.1f}C, Power: {m.power_draw_w:.1f}W")
        print("")

        # 4. AI Advisor Demo
        print("[4] AI Optimization Advisor Recommendation:")
        advisor = AIResearchAdvisor(benchmark_history=[])
        rec = advisor.get_recommendation(model_name, "NVIDIA H100 80GB", 100, 5000.0)
        print(f"  Best Runtime: {rec.best_runtime}")
        print(f"  Best Quantization: {rec.best_quantization}")
        print(f"  Best Batch Size: {rec.best_batch_size}")
        print(f"  Rationale: {rec.explanation}")
        print("")

        print("=== Demo Completed Successfully ===")

    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(run_demo())
