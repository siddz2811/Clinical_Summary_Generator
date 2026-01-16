"""
Streamlit Frontend for Clinical Summary Generator

This module provides a web interface for generating and viewing
patient clinical summaries with citations.
"""

import streamlit as st
import requests
import json
from typing import Optional, Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Clinical Summary Generator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .citation-box {
        background-color: #f0f7ff;
        border-left: 4px solid #1f77b4;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    .source-tag {
        background-color: #e8f4ea;
        color: #2e7d32;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .finding-item {
        background-color: #fff3e0;
        border-left: 3px solid #ff9800;
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 0 4px 4px 0;
    }
    .recommendation-item {
        background-color: #e3f2fd;
        border-left: 3px solid #2196f3;
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 0 4px 4px 0;
    }
    .summary-text {
        background-color: #fafafa;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        line-height: 1.8;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 15px;
        border-radius: 0 5px 5px 0;
    }
    .info-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> Dict[str, Any]:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "detail": f"Status code: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "offline", "detail": "Cannot connect to API. Make sure backend is running."}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def get_available_patients() -> list:
    """Fetch list of available patient IDs from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/patients", timeout=5)
        if response.status_code == 200:
            return response.json().get("patients", [])
        return []
    except Exception:
        return []


def get_patient_info(patient_id: int) -> Optional[Dict[str, Any]]:
    """Fetch patient information from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/patient/{patient_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def generate_summary(patient_id: int) -> Dict[str, Any]:
    """Call the API to generate a clinical summary."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate_summary",
            json={"patient_id": patient_id},
            timeout=120  # Allow up to 2 minutes for LLM response
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        elif response.status_code == 404:
            return {"success": False, "error": f"Patient ID {patient_id} not found"}
        else:
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    error_detail = error_data.get("detail", "Unknown error")
                else:
                    error_detail = str(error_data)
            except (ValueError, AttributeError):
                error_detail = response.text or "Unknown error"
            return {"success": False, "error": error_detail}
            
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to API. Make sure the backend is running."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def display_citation(claim: Dict[str, Any], index: int):
    """Display a single citation in a formatted box."""
    st.markdown(f"""
    <div class="citation-box">
        <strong>{index}. {claim.get('claim', 'N/A')}</strong><br>
        <span class="source-tag">📁 {claim.get('source_file', 'N/A')}</span>
        {f'<span class="source-tag">📅 {claim.get("source_date")}</span>' if claim.get('source_date') else ''}
        {f'<br><small style="color: #666;">ℹ️ {claim.get("source_details")}</small>' if claim.get('source_details') else ''}
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<p class="main-header">🏥 Clinical Summary Generator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate AI-powered clinical summaries with source citations</p>', unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔍 Patient Selection")
        
        # Patient ID input
        patient_id = st.number_input(
            "Enter Patient ID",
            min_value=1,
            value=1001,
            step=1,
            help="Enter the patient ID to generate summary for"
        )
        
        # Get patient info if available
        available_patients = get_available_patients()
        if available_patients:
            st.caption(f"Available: {', '.join(map(str, available_patients))}")
        
        # Display patient info
        if patient_id in available_patients:
            patient_info = get_patient_info(patient_id)
            if patient_info:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.markdown(f"**Patient ID:** {patient_info['patient_id']}")
                st.markdown(f"**Episodes:** {', '.join(map(str, patient_info['episodes']))}")
                st.markdown("**Data Available:**")
                for table, count in patient_info.get('tables', {}).items():
                    st.markdown(f"- {table}: {count} records")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate button
        st.markdown("<br>", unsafe_allow_html=True)
        generate_button = st.button(
            "🚀 Generate Summary",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        st.subheader("📋 Clinical Summary")
        
        if generate_button:
            # Check if API is available
            health = check_api_health()
            if health.get("status") not in ["healthy", "degraded"]:
                st.markdown("""
                <div class="error-box">
                    <strong>❌ API Not Available</strong><br>
                    Please start the backend server:<br>
                    <code>python main.py</code>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Generate summary with progress indicator
                with st.spinner("🔄 Generating clinical summary... This may take a moment."):
                    result = generate_summary(patient_id)
                
                if result["success"]:
                    data = result["data"]
                    
                    # Store in session state for persistence
                    st.session_state["last_summary"] = data
                    
                    st.success(f"✅ Summary generated at {data.get('generated_at', 'N/A')}")
                    
                    # Show parse error warning if present
                    if data.get("parse_error"):
                        st.warning(f"⚠️ {data['parse_error']}")
                    
                    # Tabs for different sections
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📝 Summary", 
                        "📎 Citations", 
                        "🔑 Key Findings", 
                        "💡 Recommendations"
                    ])
                    
                    with tab1:
                        st.markdown("### Narrative Summary")
                        summary_text = data.get("summary_text", "No summary available")
                        st.markdown(f'<div class="summary-text">{summary_text}</div>', unsafe_allow_html=True)
                    
                    with tab2:
                        st.markdown("### Source Citations")
                        st.markdown("Every claim in the summary is linked to source data:")
                        
                        cited_claims = data.get("cited_claims", [])
                        if cited_claims:
                            for i, claim in enumerate(cited_claims, 1):
                                display_citation(claim, i)
                        else:
                            st.info("No structured citations available. Check the summary text for inline citations.")
                    
                    with tab3:
                        st.markdown("### Key Clinical Findings")
                        findings = data.get("key_findings", [])
                        if findings:
                            for finding in findings:
                                st.markdown(f'<div class="finding-item">{finding}</div>', unsafe_allow_html=True)
                        else:
                            st.info("No key findings extracted.")
                    
                    with tab4:
                        st.markdown("### Clinical Recommendations")
                        recommendations = data.get("recommendations", [])
                        if recommendations:
                            for rec in recommendations:
                                st.markdown(f'<div class="recommendation-item">{rec}</div>', unsafe_allow_html=True)
                        else:
                            st.info("No recommendations provided.")
                    
                    # Download option
                    st.divider()
                    st.download_button(
                        label="📥 Download Summary as JSON",
                        data=json.dumps(data, indent=2),
                        file_name=f"patient_{patient_id}_summary.json",
                        mime="application/json"
                    )
                    
                else:
                    st.markdown(f"""
                    <div class="error-box">
                        <strong>❌ Error Generating Summary</strong><br>
                        {result.get('error', 'Unknown error')}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show last summary if available
        elif "last_summary" in st.session_state:
            data = st.session_state["last_summary"]
            st.info(f"Showing previous summary for Patient {data.get('patient_id')} (generated at {data.get('generated_at', 'N/A')})")
            
            # Display tabs with previous data
            tab1, tab2, tab3, tab4 = st.tabs([
                "📝 Summary", 
                "📎 Citations", 
                "🔑 Key Findings", 
                "💡 Recommendations"
            ])
            
            with tab1:
                st.markdown("### Narrative Summary")
                summary_text = data.get("summary_text", "No summary available")
                st.markdown(f'<div class="summary-text">{summary_text}</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown("### Source Citations")
                cited_claims = data.get("cited_claims", [])
                if cited_claims:
                    for i, claim in enumerate(cited_claims, 1):
                        display_citation(claim, i)
                else:
                    st.info("No structured citations available.")
            
            with tab3:
                st.markdown("### Key Clinical Findings")
                findings = data.get("key_findings", [])
                if findings:
                    for finding in findings:
                        st.markdown(f'<div class="finding-item">{finding}</div>', unsafe_allow_html=True)
                else:
                    st.info("No key findings extracted.")
            
            with tab4:
                st.markdown("### Clinical Recommendations")
                recommendations = data.get("recommendations", [])
                if recommendations:
                    for rec in recommendations:
                        st.markdown(f'<div class="recommendation-item">{rec}</div>', unsafe_allow_html=True)
                else:
                    st.info("No recommendations provided.")
        else:
            st.info("👆 Enter a Patient ID and click 'Generate Summary' to get started.")


if __name__ == "__main__":
    main()
