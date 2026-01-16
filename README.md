# Clinical Summary Generator — Windows

Prerequisites
- Python 3.8+ installed and available as `python`.
- Git (optional, for cloning).
- Internet access for installing packages and calling the LLM service.

Setup (Windows)

1. Clone the repo
```powershell
git clone https://github.com/siddz2811/Clinical_Summary_Generator.git
cd Clinical_Summary_Generator
```

2. Create and activate a virtual environment
python -m venv myenv
myenv\Scripts\activate

3. Install dependencies
- If you have a `requirements.txt`:
pip install -r requirements.txt


4. Set the LLM API key in an .env file
```
GROQ_API_KEY=your_groq_api_key_here
```
Run

1. Start the backend (FastAPI)
```cmd
python -m uvicorn main:app --reload --port 8000
```
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

2. Start the frontend (Streamlit)
```cmd
streamlit run app.py 
```
Brief file explanations
- main.py — FastAPI server and endpoints (/, /health, /patients, /patient/{id}, /generate_summary). Starts uvicorn when run.
- app.py — Streamlit frontend that calls the API and displays summaries.
- data_load.py — PatientDatabase: loads CSVs into pandas DataFrames and provides query/filter helpers.
- clinical_summary.py — ClinicalSummaryGenerator: builds patient context and calls the LLM (Groq via LangChain) to produce summaries.
