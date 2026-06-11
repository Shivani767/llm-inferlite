from fastapi import APIRouter, HTTPException
from services.simulator import ServingSimulator, SimulationConfig, SimulationResult, DistributedInferenceSimulator, DistributedSimulationConfig, DistributedSimulationResult
from typing import Optional

router = APIRouter()

@router.post("/simulate", response_model=SimulationResult)
async def run_serving_simulation(config: SimulationConfig):
    """
    Run a discrete-event simulation of model serving performance.
    Helpful for capacity planning and understanding bottleneck under load.
    """
    try:
        simulator = ServingSimulator(config)
        result = await simulator.run_simulation()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/distributed-simulate", response_model=DistributedSimulationResult)
async def run_distributed_simulation(config: DistributedSimulationConfig):
    """
    Run a distributed inference simulation with Tensor Parallelism (TP),
    Pipeline Parallelism (PP), or Hybrid TP+PP.
    """
    try:
        simulator = DistributedInferenceSimulator(config)
        result = await simulator.run_simulation()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/architectures")
async def list_simulated_architectures():
    """
    List predefined serving architectures for simulation.
    """
    return [
        {"id": "single_node", "name": "Single Node (GPU)", "nodes": 1, "concurrency": 32},
        {"id": "multi_node_lb", "name": "Multi-Node Load Balanced", "nodes": 4, "concurrency": 32},
        {"id": "autoscaling", "name": "Autoscaling Cluster (1-10 nodes)", "nodes": 1, "is_autoscaling": True}
    ]

@router.get("/quantization-methods")
async def list_quantization_methods():
    """
    List all supported quantization methods.
    """
    return [
        {"id": "gptq", "name": "GPTQ", "precision": "INT4", "description": "Generalized Post-Training Quantization"},
        {"id": "awq", "name": "AWQ", "precision": "INT4", "description": "Activation-aware Weight Quantization"},
        {"id": "bitsandbytes", "name": "BitsAndBytes", "precision": "INT8", "description": "LLM.int8() / QLoRA"},
        {"id": "gguf", "name": "GGUF", "precision": "Q4_K_M", "description": "llama.cpp quantization format"},
        {"id": "squeezellm", "name": "SqueezeLLM", "precision": "INT4", "description": "Density-aware quantization"},
        {"id": "dynamic_int8", "name": "Dynamic INT8", "precision": "INT8", "description": "Dynamic activation quantization"}
    ]
