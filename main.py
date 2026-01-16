"""
FastAPI Backend for Clinical Summary Generator

This module provides a REST API for generating patient clinical summaries
with citations using the ClinicalSummaryGenerator.
"""

import os
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from clinical_summary import ClinicalSummaryGenerator
from data_load import PatientDatabase

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Summary API",
    description="API for generating patient clinical summaries with LLM and citations",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the generator (lazy loading)
_generator: Optional[ClinicalSummaryGenerator] = None
_database: Optional[PatientDatabase] = None

# Thread pool executor for running blocking operations
_executor = ThreadPoolExecutor(max_workers=4)


def get_generator() -> ClinicalSummaryGenerator:
    """Get or create the ClinicalSummaryGenerator instance."""
    global _generator
    if _generator is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY environment variable not set"
            )
        _generator = ClinicalSummaryGenerator(data_dir="data")
    return _generator


def get_database() -> PatientDatabase:
    """Get or create the PatientDatabase instance."""
    global _database
    if _database is None:
        _database = PatientDatabase(data_dir="data")
    return _database


# Pydantic models for request/response
class SummaryRequest(BaseModel):
    """Request model for summary generation."""
    patient_id: int = Field(..., description="The patient ID to generate summary for", examples=[1001])
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "patient_id": 1001
            }
        }
    }


class CitedClaim(BaseModel):
    """Model for a cited claim in the summary."""
    claim: str = Field(..., description="The clinical statement")
    source_file: str = Field(..., description="Source CSV file")
    source_date: Optional[str] = Field(None, description="Date of the source data")
    source_details: Optional[str] = Field(None, description="Additional context")


class SummaryResponse(BaseModel):
    """Response model for the generated summary."""
    patient_id: int
    summary_text: str = Field(..., description="The narrative summary with inline citations")
    cited_claims: List[CitedClaim] = Field(default=[], description="List of claims with sources")
    key_findings: List[str] = Field(default=[], description="Key clinical findings")
    recommendations: List[str] = Field(default=[], description="Clinical recommendations")
    generated_at: str = Field(..., description="Timestamp of generation")
    parse_error: Optional[str] = Field(None, description="Error message if JSON parsing failed")


class PatientInfo(BaseModel):
    """Model for patient information."""
    patient_id: int
    episodes: List[int]
    tables: Dict[str, int]  # table name -> row count


class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str
    api_key_configured: bool
    database_loaded: bool
    available_patients: List[int]


# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Clinical Summary API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "POST /generate_summary": "Generate clinical summary for a patient",
            "GET /patients": "List all available patients",
            "GET /patient/{patient_id}": "Get patient information",
            "GET /health": "Health check"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    api_key_configured = bool(os.environ.get("GROQ_API_KEY"))
    
    try:
        db = get_database()
        patients = [int(p) for p in db.get_all_patient_ids()]
        database_loaded = True
    except Exception:
        patients = []
        database_loaded = False
    
    return HealthResponse(
        status="healthy" if api_key_configured and database_loaded else "degraded",
        api_key_configured=api_key_configured,
        database_loaded=database_loaded,
        available_patients=patients
    )


@app.get("/patients", tags=["Patients"])
async def list_patients():
    """List all available patient IDs."""
    try:
        db = get_database()
        patients = [int(p) for p in db.get_all_patient_ids()]
        return {
            "patients": patients,
            "count": len(patients)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patient/{patient_id}", response_model=PatientInfo, tags=["Patients"])
async def get_patient_info(patient_id: int):
    """Get information about a specific patient."""
    try:
        db = get_database()
        
        # Check if patient exists
        all_patients = [int(p) for p in db.get_all_patient_ids()]
        if patient_id not in all_patients:
            raise HTTPException(
                status_code=404,
                detail=f"Patient ID {patient_id} not found. Available patients: {all_patients}"
            )
        
        # Get patient data
        patient_data = db.filter_by_patient(patient_id)
        episodes = [int(e) for e in db.get_episodes_for_patient(patient_id)]
        
        tables = {name: len(df) for name, df in patient_data.items()}
        
        return PatientInfo(
            patient_id=patient_id,
            episodes=episodes,
            tables=tables
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_summary", response_model=SummaryResponse, tags=["Summary"])
async def generate_summary(request: SummaryRequest):
    """
    Generate a clinical summary for a patient.
    
    This endpoint accepts a patient_id and returns a comprehensive clinical
    summary with citations linking each claim to source data from the CSVs.
    
    **Example Request:**
    ```json
    {
        "patient_id": 1001
    }
    ```
    
    **Example Response:**
    ```json
    {
        "patient_id": 1001,
        "summary_text": "Patient overview...",
        "cited_claims": [...],
        "key_findings": [...],
        "recommendations": [...],
        "generated_at": "2024-01-01T12:00:00"
    }
    ```
    
    **Note:** This operation may take 30-120 seconds as it involves LLM processing.
    """
    try:
        generator = get_generator()
        
        # Check if patient exists (this is quick, can be done synchronously)
        all_patients = [int(p) for p in generator.db.get_all_patient_ids()]
        if request.patient_id not in all_patients:
            raise HTTPException(
                status_code=404,
                detail=f"Patient ID {request.patient_id} not found. Available patients: {all_patients}"
            )
        
        # Run the blocking LLM operation in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            generator.generate_summary,
            request.patient_id
        )
        
        # Extract the summary (it's a string from the LLM)
        summary_text = result.get("summary", "")
        
        # Build response
        # Note: The current implementation returns a plain string summary.
        # If you want structured citations, key_findings, and recommendations,
        # you'll need to parse the LLM output or modify the prompt to return JSON.
        response = SummaryResponse(
            patient_id=request.patient_id,
            summary_text=summary_text if isinstance(summary_text, str) else str(summary_text),
            cited_claims=[],  # Empty for now - would need parsing logic
            key_findings=[],  # Empty for now - would need parsing logic
            recommendations=[],  # Empty for now - would need parsing logic
            generated_at=result.get("generated_at", datetime.now().isoformat()),
            parse_error=None
        )
        
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


@app.get("/patient/{patient_id}/context", tags=["Patients"])
async def get_patient_context(patient_id: int):
    """Get the raw patient context that would be sent to the LLM."""
    try:
        generator = get_generator()
        
        # Check if patient exists
        all_patients = [int(p) for p in generator.db.get_all_patient_ids()]
        if patient_id not in all_patients:
            raise HTTPException(
                status_code=404,
                detail=f"Patient ID {patient_id} not found"
            )
        
        context = generator.build_patient_context(patient_id)
        
        return {
            "patient_id": patient_id,
            "context": context
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    
    print("Starting Clinical Summary API...")
    print("API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
