import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent import FacultyBylawsAgent
from dotenv import load_dotenv

# Add the project root folder to the Python path (if needed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Initialize the AI agent
agent = FacultyBylawsAgent("data/English translation.pdf")

# Create the FastAPI app
app = FastAPI(
    title="Faculty Bylaws Agent API",
    description="API for asking questions to the Faculty Bylaws Agent powered by Langchain & DeepSeek.",
    version="1.0.0"
)

# Define the request body using Pydantic
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask", response_model=dict)
async def ask(request: QuestionRequest):
    """
    Ask the Faculty Bylaws Agent a question.
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="No question provided")
    answer = agent.answer_question(request.question)
    return {"answer": answer}


# Run the FastAPI app using Uvicorn (instead of Flask's app.run)
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
