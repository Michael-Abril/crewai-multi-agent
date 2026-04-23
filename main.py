import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

AKASH_API_KEY = os.environ.get("AKASH_API_KEY", "")
AKASH_BASE_URL = "https://chatapi.akash.network/api/v1"
MODEL = "Meta-Llama-3-3-70B-Instruct"

# In-memory job store
jobs: dict = {}


def make_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=MODEL,
        api_key=AKASH_API_KEY,
        base_url=AKASH_BASE_URL,
        max_tokens=1024,
        temperature=0.3,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not AKASH_API_KEY:
        # Continue startup — /health will return 503 until key is available
        print("WARNING: AKASH_API_KEY not set. Crew runs will fail until configured.")
    yield


app = FastAPI(title="CrewAI Multi-Agent API", lifespan=lifespan)


class RunRequest(BaseModel):
    topic: str


def _run_crew(job_id: str, topic: str) -> None:
    from crewai import Agent, Crew, Task

    try:
        jobs[job_id]["status"] = "running"
        start = time.time()

        llm = make_llm()

        researcher = Agent(
            role="Researcher",
            goal=f"Gather key facts about: {topic}",
            backstory="You are a concise researcher who extracts the most important facts on any topic.",
            llm=llm,
            max_iter=3,
            verbose=False,
            allow_delegation=False,
        )

        writer = Agent(
            role="Writer",
            goal=f"Write a clear, multi-paragraph report about: {topic}",
            backstory="You are a technical writer who produces well-structured, jargon-free reports.",
            llm=llm,
            max_iter=3,
            verbose=False,
            allow_delegation=False,
        )

        research_task = Task(
            description=(
                f"Research the topic: '{topic}'. "
                "Produce exactly 5 bullet points, each one sentence, covering the most important facts."
            ),
            expected_output="5 concise bullet points about the topic.",
            agent=researcher,
        )

        write_task = Task(
            description=(
                f"Using the research notes, write a 3-paragraph report about: '{topic}'. "
                "Each paragraph must be at least 3 sentences. Do not use bullet points."
            ),
            expected_output="A 3-paragraph prose report.",
            agent=writer,
            context=[research_task],
        )

        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            verbose=False,
        )

        result = crew.kickoff()
        duration = round(time.time() - start, 2)

        jobs[job_id].update(
            {
                "status": "done",
                "result": str(result),
                "duration_seconds": duration,
            }
        )
    except Exception as exc:
        jobs[job_id].update({"status": "error", "error": str(exc)})


@app.get("/health")
def health():
    if not AKASH_API_KEY:
        return JSONResponse(
            status_code=503, content={"status": "error", "detail": "AKASH_API_KEY not configured"}
        )
    return {"status": "ok"}


@app.get("/agents")
def list_agents():
    return {
        "agents": [
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
    }


@app.post("/run")
def run_crew(request: RunRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "topic": request.topic}
    background_tasks.add_task(_run_crew, job_id, request.topic)
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
