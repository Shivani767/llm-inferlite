from fastapi import APIRouter, HTTPException
from services.reports import ResearchReportGenerator as SimpleGenerator
from reports.generator import ResearchReportGenerator
from typing import Dict, Any

router = APIRouter()

@router.post("/generate")
async def generate_report(data: Dict[str, Any]):
    """
    Module 14: Generate a comprehensive research report.
    Consolidates benchmarks, quantization deltas, and cost analysis.
    """
    try:
        generator = ResearchReportGenerator(data)
        markdown_content = generator.generate_markdown()
        return {
            "report_md": markdown_content,
            "report_json": generator.export_to_json()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
