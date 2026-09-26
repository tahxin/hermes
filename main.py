import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Connects to OpenAI, Groq, OpenRouter, or Hermes-compatible APIs
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "dummy-key"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

class PromptRequest(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body style="font-family: Arial; text-align: center; margin-top: 50px;">
            <h2>AI Agent is Live on Render!</h2>
            <p>Send a POST request to <code>/chat</code> with <code>{"prompt": "Your question"}</code></p>
        </body>
    </html>
    """

@app.post("/chat")
def generate_chat(req: PromptRequest):
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful, direct AI agent."},
            {"role": "user", "content": req.prompt}
        ],
    )
    return {"reply": response.choices[0].message.content}
