"""
Multi-agent research + writing pipeline using Akash ML API (Llama 3.3 70B).
Two sequential agents: Researcher → Writer. Uses stdlib urllib for LLM calls.
No extra packages beyond fastapi+uvicorn — avoids pip install timeouts on Akash.
"""
import json
import os
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

AKASH_API_KEY = os.environ.get("AKASH_API_KEY", "")
AKASH_BASE_URL = "https://chatapi.akash.network/api/v1"
MODEL = "Meta-Llama-3-3-70B-Instruct"

jobs: dict = {}

AGENTS_LIST = [
    {
        "name": "Researcher",
        "role": "Gathers key facts about the requested topic",
        "model": MODEL,
    },
    {
        "name": "Writer",
        "role": "Synthesizes research into a clear multi-paragraph report",
        "model": MODEL,
    },
]


def llm_call(system: str, user: str, max_tokens: int = 512) -> str:
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
    ).encode()

    req = urllib.request.Request(
        f"{AKASH_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AKASH_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def _run_pipeline(job_id: str, topic: str) -> None:
    try:
        jobs[job_id]["status"] = "running"
        start = time.time()

        # Agent 1 — Researcher
        research = llm_call(
            system=(
                "You are a concise researcher. Given a topic, produce exactly 5 bullet points "
                "covering the most important facts. Each bullet is one clear sentence."
            ),
            user=f"Research topic: {topic}",
            max_tokens=400,
        )

        # Agent 2 — Writer (receives Researcher output as context)
        report = llm_call(
            system=(
                "You are a technical writer. Given research notes, write a 3-paragraph report. "
                "Each paragraph must be at least 3 sentences. Use prose, not bullet points."
            ),
            user=f"Research notes:\n{research}\n\nWrite a report about: {topic}",
            max_tokens=800,
        )

        jobs[job_id].update(
            {
                "status": "done",
                "result": report,
                "research_notes": research,
                "duration_seconds": round(time.time() - start, 2),
            }
        )
    except Exception as exc:
        jobs[job_id].update({"status": "error", "error": str(exc)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not AKASH_API_KEY:
        print("WARNING: AKASH_API_KEY not set. LLM calls will fail until configured.")
    yield


app = FastAPI(title="Multi-Agent Research API", lifespan=lifespan)


class RunRequest(BaseModel):
    topic: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/agents")
def list_agents():
    return {"agents": AGENTS_LIST}


@app.post("/run")
def run_pipeline(request: RunRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "topic": request.topic}
    background_tasks.add_task(_run_pipeline, job_id, request.topic)
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 3000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
