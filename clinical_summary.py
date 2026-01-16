"""
Clinical Summary Generator

This module generates clinical summaries for patients using LLM integration.
It fetches patient data from the data layer and uses LangChain with Groq
to generate comprehensive clinical narratives.
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from data_load import PatientDatabase


class ClinicalSummaryGenerator:
    """
    Generates clinical summaries for patients using LLM.
    
    This class fetches patient data from the data layer, formats it into
    a clinical context, and uses Groq LLM to generate comprehensive summaries.
    """
    
    # System prompt for the LLM - acts as a home health clinician
    SYSTEM_PROMPT = """You are an experienced home health clinician reviewing patient records. 
Your task is to generate a comprehensive clinical summary that tells the story of the patient's 
current condition. Write in a professional, clinical tone suitable for care coordination.

Focus on:
1. **Primary Diagnoses**: Identify and explain the main medical conditions
2. **Recent Vital Sign Trends**: Analyze patterns, flag any concerning values
3. **Active Wounds**: Describe wound status, healing progress, and care needs
4. **Medications**: Review current regimen, note any concerns or interactions
5. **Functional Status (OASIS)**: Summarize the patient's ability to perform daily activities
6. **Clinical Notes**: Extract key observations from recent nursing notes

Provide actionable insights and highlight any areas requiring attention or follow-up.
Keep the summary concise but thorough - aim for 300-500 words."""

    HUMAN_PROMPT = """Please generate a clinical summary for the following patient:

{patient_context}

Generate a comprehensive clinical summary based on the above data."""

    def __init__(
        self, 
        data_dir: str = "data",
        groq_api_key: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile"
    ):
        """
        Initialize the Clinical Summary Generator.
        
        Args:
            data_dir: Path to the data directory
            groq_api_key: Groq API key (or set GROQ_API_KEY env variable)
            model_name: Name of the Groq model to use
        """
        # Initialize the database
        self.db = PatientDatabase(data_dir)
        
        # Set up Groq API key - use parameter if provided, otherwise check environment variable
        self.api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY environment variable "
                "or pass groq_api_key parameter."
            )
        
        # Initialize the LLM
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=model_name,
            temperature=0.3,  # Lower temperature for more consistent clinical output
            max_tokens=2048
        )
        
        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", self.HUMAN_PROMPT)
        ])
        
        # Create the chain
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _format_diagnoses(self, df: pd.DataFrame) -> str:
        """Format diagnoses data into readable text."""
        if df.empty:
            return "No diagnoses recorded."
        
        # Get unique diagnoses
        unique_diagnoses = df[['diagnosis_description', 'diagnosis_code']].drop_duplicates()
        
        lines = ["DIAGNOSES:"]
        for _, row in unique_diagnoses.iterrows():
            code = row['diagnosis_code'] if row['diagnosis_code'] != 'Unknown' else 'N/A'
            lines.append(f"  - {row['diagnosis_description']} (Code: {code})")
        
        return "\n".join(lines)
    
    def _format_medications(self, df: pd.DataFrame) -> str:
        """Format medications data into readable text."""
        if df.empty:
            return "No medications recorded."
        
        # Get unique medications
        unique_meds = df[['medication_name', 'frequency', 'classification', 'reason']].drop_duplicates()
        
        lines = ["CURRENT MEDICATIONS:"]
        
        # Group by classification
        for classification in unique_meds['classification'].unique():
            class_meds = unique_meds[unique_meds['classification'] == classification]
            lines.append(f"\n  [{classification}]")
            for _, row in class_meds.iterrows():
                reason = row['reason'].strip() if pd.notna(row['reason']) else 'N/A'
                lines.append(f"    - {row['medication_name']}")
                lines.append(f"      Frequency: {row['frequency']}, Reason: {reason}")
        
        return "\n".join(lines)
    
    def _format_vitals(self, df: pd.DataFrame) -> str:
        """Format vitals data into readable text with trends."""
        if df.empty:
            return "No vital signs recorded."
        
        lines = ["VITAL SIGNS:"]
        
        # Sort by date
        df_sorted = df.sort_values('visit_date', ascending=False)
        
        # Get date range
        earliest = df_sorted['visit_date'].min()
        latest = df_sorted['visit_date'].max()
        earliest_str = earliest.strftime('%Y-%m-%d') if pd.notna(earliest) else 'N/A'
        latest_str = latest.strftime('%Y-%m-%d') if pd.notna(latest) else 'N/A'
        lines.append(f"  Period: {earliest_str} to {latest_str}")
        
        # Analyze each vital type
        for vital_type in df_sorted['vital_type'].unique():
            vital_data = df_sorted[df_sorted['vital_type'] == vital_type]
            
            readings = vital_data['reading'].dropna()
            if len(readings) == 0:
                continue
            
            # Get latest reading from the first row (already sorted by date descending)
            # Since vital_data is sorted by date descending, the first non-NaN reading is the latest
            latest_reading = readings.iloc[0]
            min_val = vital_data['min_value'].iloc[0] if pd.notna(vital_data['min_value'].iloc[0]) else None
            max_val = vital_data['max_value'].iloc[0] if pd.notna(vital_data['max_value'].iloc[0]) else None
            
            # Calculate statistics
            avg_reading = readings.mean()
            
            # Check if out of range
            status = "Normal"
            if min_val and latest_reading < min_val:
                status = "LOW"
            elif max_val and latest_reading > max_val:
                status = "HIGH"
            
            range_str = ""
            if min_val or max_val:
                range_str = f" [Range: {min_val or 'N/A'}-{max_val or 'N/A'}]"
            
            lines.append(f"\n  {vital_type}:")
            lines.append(f"    Latest: {latest_reading} ({status}){range_str}")
            lines.append(f"    Average: {avg_reading:.1f} over {len(readings)} readings")
        
        return "\n".join(lines)
    
    def _format_wounds(self, df: pd.DataFrame) -> str:
        """Format wounds data into readable text."""
        if df.empty:
            return "No active wounds recorded."
        
        lines = ["ACTIVE WOUNDS:"]
        
        # Get unique wounds by location
        unique_wounds = df[['description', 'location', 'onset_date']].drop_duplicates()
        
        for _, row in unique_wounds.iterrows():
            onset = row['onset_date'].strftime('%Y-%m-%d') if pd.notna(row['onset_date']) else 'Unknown'
            lines.append(f"\n  Location: {row['location']}")
            lines.append(f"    Type: {row['description']}")
            lines.append(f"    Onset Date: {onset}")
            
            # Get visit history for this wound
            wound_visits = df[df['location'] == row['location']]['visit_date'].sort_values()
            if len(wound_visits) > 0:
                lines.append(f"    Visits: {len(wound_visits)} wound care visits recorded")
                latest_visit = wound_visits.iloc[-1]
                latest_visit_str = latest_visit.strftime('%Y-%m-%d') if pd.notna(latest_visit) else 'N/A'
                lines.append(f"    Latest visit: {latest_visit_str}")
        
        return "\n".join(lines)
    
    def _format_oasis(self, df: pd.DataFrame) -> str:
        """Format OASIS functional assessment data."""
        if df.empty:
            return "No functional assessments recorded."
        
        lines = ["FUNCTIONAL STATUS (OASIS Assessment):"]
        
        # Get the most recent assessment
        df_sorted = df.sort_values('assessment_date', ascending=False)
        latest = df_sorted.iloc[0]
        
        lines.append(f"  Assessment Date: {latest['assessment_date'].strftime('%Y-%m-%d') if pd.notna(latest['assessment_date']) else 'N/A'}")
        lines.append(f"  Assessment Type: {latest['assessment_type']}")
        
        # Extract functional levels (just the number prefix for clarity)
        def extract_level(text):
            if pd.isna(text):
                return "N/A"
            # Return the full text for context
            return text.strip()
        
        lines.append(f"\n  Grooming: {extract_level(latest['grooming'])}")
        lines.append(f"\n  Bathing: {extract_level(latest['bathing'])}")
        lines.append(f"\n  Toilet Transfer: {extract_level(latest['toilet_transfer'])}")
        lines.append(f"\n  Transfer: {extract_level(latest['transfer'])}")
        lines.append(f"\n  Ambulation: {extract_level(latest['ambulation'])}")
        
        return "\n".join(lines)
    
    def _format_notes(self, df: pd.DataFrame) -> str:
        """Format clinical notes data."""
        if df.empty:
            return "No clinical notes recorded."
        
        lines = ["RECENT CLINICAL NOTES:"]
        
        # Sort by date and get recent notes
        df_sorted = df.sort_values('note_date', ascending=False)
        
        # Limit to most recent 5 notes to avoid context overload
        recent_notes = df_sorted.head(5)
        
        for _, row in recent_notes.iterrows():
            date_str = row['note_date'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['note_date']) else 'N/A'
            lines.append(f"\n  [{date_str}] {row['note_type']}:")
            
            # Truncate very long notes
            note_text = str(row['note_text'])
            if len(note_text) > 500:
                note_text = note_text[:500] + "..."
            lines.append(f"    {note_text}")
        
        return "\n".join(lines)
    
    def build_patient_context(self, patient_id: int) -> str:
        """
        Build a comprehensive context string for a patient.
        
        Args:
            patient_id: The patient ID to build context for
            
        Returns:
            Formatted string containing all patient data
        """
        # Verify patient exists
        if patient_id not in self.db.get_all_patient_ids():
            raise ValueError(f"Patient ID {patient_id} not found in database.")
        
        # Get all patient data
        patient_data = self.db.filter_by_patient(patient_id)
        
        # Get episodes
        episodes = self.db.get_episodes_for_patient(patient_id)
        
        # Build context sections
        sections = [
            f"PATIENT ID: {patient_id}",
            f"EPISODE(S): {', '.join(map(str, episodes))}",
            f"REPORT GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
            self._format_diagnoses(patient_data.get('diagnoses', pd.DataFrame())),
            "",
            self._format_medications(patient_data.get('medications', pd.DataFrame())),
            "",
            self._format_vitals(patient_data.get('vitals', pd.DataFrame())),
            "",
            self._format_wounds(patient_data.get('wounds', pd.DataFrame())),
            "",
            self._format_oasis(patient_data.get('oasis', pd.DataFrame())),
            "",
            self._format_notes(patient_data.get('notes', pd.DataFrame())),
        ]
        
        return "\n".join(sections)
    
    def generate_summary(self, patient_id: int) -> Dict[str, Any]:
        """
        Generate a clinical summary for a patient.
        
        Args:
            patient_id: The patient ID to generate summary for
            
        Returns:
            Dictionary containing the patient context and generated summary
        """
        # Build the patient context
        patient_context = self.build_patient_context(patient_id)
        
        # Generate the summary using the LLM
        summary = self.chain.invoke({"patient_context": patient_context})
        
        return {
            "patient_id": patient_id,
            "context": patient_context,
            "summary": summary,
            "generated_at": datetime.now().isoformat()
        }
    
    def generate_summary_only(self, patient_id: int) -> str:
        """
        Generate just the clinical summary text for a patient.
        
        Args:
            patient_id: The patient ID to generate summary for
            
        Returns:
            The generated clinical summary text
        """
        result = self.generate_summary(patient_id)
        return result["summary"]


def generate_clinical_summary(
    patient_id: int,
    data_dir: str = "data",
    groq_api_key: Optional[str] = None
) -> str:
    """
    Convenience function to generate a clinical summary for a patient.
    
    Args:
        patient_id: The patient ID to generate summary for
        data_dir: Path to the data directory
        groq_api_key: Groq API key (optional if GROQ_API_KEY env var is set)
        
    Returns:
        The generated clinical summary text
    """
    generator = ClinicalSummaryGenerator(
        data_dir=data_dir,
        groq_api_key=groq_api_key
    )
    return generator.generate_summary_only(patient_id)


# Example usage and demonstration
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Clinical Summary Generator")
    print("=" * 60)
    
    # Check for API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\nError: GROQ_API_KEY environment variable not set.")
        print("Please set your Groq API key:")
        print("  Windows: set GROQ_API_KEY=your_api_key_here")
        print("  Linux/Mac: export GROQ_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Get patient ID from command line or use default
    patient_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1001
    
    print(f"\nGenerating clinical summary for Patient {patient_id}...")
    print("-" * 60)
    
    try:
        # Initialize generator
        generator = ClinicalSummaryGenerator(data_dir="data")
        
        # Build and display context
        print("\n[PATIENT CONTEXT]")
        print("=" * 60)
        context = generator.build_patient_context(patient_id)
        print(context)
        
        # Generate summary
        print("\n[GENERATING CLINICAL SUMMARY...]")
        print("=" * 60)
        result = generator.generate_summary(patient_id)
        
        print("\n[CLINICAL SUMMARY]")
        print("=" * 60)
        print(result["summary"])
        
        print("\n" + "=" * 60)
        print(f"Summary generated at: {result['generated_at']}")
        
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError generating summary: {e}")
        sys.exit(1)
