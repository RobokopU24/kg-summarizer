# kg-summarizer
Knowledge graph summarization using LLMs with a FastAPI interface

# Create Environment
python -m venv venv
venv\Scripts\activate
pip install -r .\requirements.txt

# Start FastAPI/Uvicorn Server
uvicorn main:app --reload

# Server Update Steps
1) Push changes and make new release on github
    - Make sure release version matches FastAPI docs version in kg_summarizer/server.py and setup.py
2) Update Helm chart
    - Update values.yaml with new release version
    - cd translator-devops/helm/kg-summarizer
    - helm -n translator-dev upgrade -f values-populated.yaml kg-summarizer .
