import time, os, uuid, json
from pathlib import Path
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
import torch
from PIL import Image
from .settings import settings

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "diffusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CFG = json.loads((ROOT/"week7_run_config.json").read_text())

MODEL_ID = CFG["model_id"]
NEGATIVE = CFG["negative_prompt"]
SD_STEPS = int(CFG["sd_steps"])
SD_GUIDE = float(CFG["sd_guidance"])
W, H = CFG["size"]
GUARD = CFG["guardrails"]

_sd_pipe = None
def get_sd():
    global _sd_pipe
    if _sd_pipe is None:
        from diffusers import StableDiffusionPipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _sd_pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if device=="cuda" else torch.float32
        )
        if device == "cuda":
            _sd_pipe.enable_attention_slicing()
            _sd_pipe.to(device)
    return _sd_pipe

def guardrails(text: str):
    for bad in GUARD.get("unsafe_terms", []):
        if bad.lower() in text.lower():
            raise HTTPException(status_code=400, detail=f"Blocked by guardrails: {bad}")

def require_auth(authorization: str | None = Header(default=None)):
    token = settings.app_api_token
    if not token:  # auth disabled if not set
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if authorization.split(" ", 1)[1].strip() != token:
        raise HTTPException(status_code=403, detail="Invalid token")

# ----- Schemas
class QARequest(BaseModel):
    query: str = Field(...)

class QAResponse(BaseModel):
    answer: str
    citations: List[Dict[str,str]]
    latency_ms: int

class GenRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = NEGATIVE
    steps: Optional[int] = SD_STEPS
    guidance: Optional[float] = SD_GUIDE
    height: Optional[int] = H
    width: Optional[int] = W

class GenResponse(BaseModel):
    filename: str
    latency_ms: int

class AgentRequest(BaseModel):
    input: str

class AgentHop(BaseModel):
    tool: str
    input: str
    output_preview: str
    latency_ms: int

class AgentResponse(BaseModel):
    final: str
    hops: List[AgentHop]
    citations: List[Dict[str,str]] = []
    image_filename: Optional[str] = None
    latency_ms: int

app = FastAPI(title="Week 7 Backend", version="1.0")

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID}

# ----- Stub QA (replace with your Graph-RAG/Multi-Hop)
def project_qa(query: str):
    # If you call an external API, read keys from `settings.openai_api_key` / `settings.google_api_key`.
    ans = f"Stubbed answer for: {query} (replace with your Graph-RAG/Multi-Hop)."
    cites = [
        {"title": "Week 7 Repo", "url": "https://github.com/sudhakarjerribanda/Capston_Handson/tree/main/Week_7"},
        {"title": "Task 1 Colab", "url": "https://colab.research.google.com/drive/1yZd5zUUmNjeDH-_mpVEkNCO4fonsmW69?usp=sharing"}
    ]
    return ans, cites

@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest, _=require_auth()):
    t0 = time.time()
    guardrails(req.query)
    ans, cites = project_qa(req.query)
    return QAResponse(answer=ans, citations=cites, latency_ms=int((time.time()-t0)*1000))

@app.post("/generate", response_model=GenResponse)
def generate(req: GenRequest, _=require_auth()):
    t0 = time.time()
    guardrails(req.prompt)
    pipe = get_sd()
    img = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt or NEGATIVE,
        num_inference_steps=req.steps or SD_STEPS,
        guidance_scale=req.guidance or SD_GUIDE,
        height=req.height or H,
        width=req.width or W
    ).images[0]
    fname = f"sd_{uuid.uuid4().hex[:8]}.png"
    img.save(OUT_DIR/fname)
    return GenResponse(filename=fname, latency_ms=int((time.time()-t0)*1000))

@app.post("/agent", response_model=AgentResponse)
def agent(req: AgentRequest, _=require_auth()):
    t0 = time.time()
    guardrails(req.input)
    hops = []
    keywords = ["image","diagram","illustration","infographic","generate","sd"]
    if any(k in req.input.lower() for k in keywords):
        g0 = time.time()
        prompt = req.input + ", flat vector, teal+indigo accents, clean typography, rounded cards, subtle shadows"
        pipe = get_sd()
        img = pipe(
            prompt=prompt, negative_prompt=NEGATIVE,
            num_inference_steps=SD_STEPS, guidance_scale=SD_GUIDE,
            height=H, width=W
        ).images[0]
        fname = f"sd_{uuid.uuid4().hex[:8]}.png"
        img.save(OUT_DIR/fname)
        hops.append(AgentHop(tool="stable_diffusion", input=prompt, output_preview=fname, latency_ms=int((time.time()-g0)*1000)))
        final = f"Generated image {fname} for: {req.input}"
        cites = []
        image = fname
    else:
        q0 = time.time()
        ans, cites = project_qa(req.input)
        hops.append(AgentHop(tool="project_qa", input=req.input, output_preview=ans[:120], latency_ms=int((time.time()-q0)*1000)))
        final = ans
        image = None

    # Append tiny metric
    try:
        mpath = ROOT/"metrics.json"
        rec = {"ts": int(time.time()), "input": req.input, "latency_ms": int((time.time()-t0)*1000),
               "hops": [h.model_dump() for h in hops], "image": image}
        data = json.loads(mpath.read_text()) if mpath.exists() else []
        data.append(rec); mpath.write_text(json.dumps(data, indent=2))
    except Exception:
        pass

    return AgentResponse(final=final, hops=hops, citations=cites, image_filename=image, latency_ms=int((time.time()-t0)*1000))
