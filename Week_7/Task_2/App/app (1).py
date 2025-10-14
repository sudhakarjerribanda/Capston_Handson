import os, json, requests
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Week 7 App", page_icon="🧪", layout="wide")

def safe_secret(key: str, default: str = "") -> str:
    # Prefer environment (works in Colab), then secrets if available.
    val = os.environ.get(key)
    if val: 
        return val
    try:
        # Only try st.secrets if a secrets file exists
        # (Streamlit raises FileNotFoundError when no secrets.toml)
        _ = st.secrets  # access to trigger load; will raise if missing
        return st.secrets.get(key, default)
    except Exception:
        return default

# Resolve backend URL/token with robust fallback order:
# 1) ENV  2) secrets.toml (if present)  3) BACKEND_URL.txt  4) local default
BACKEND_URL   = safe_secret("BACKEND_URL", "")
APP_API_TOKEN = safe_secret("APP_API_TOKEN", "")

if not BACKEND_URL and Path("BACKEND_URL.txt").exists():
    BACKEND_URL = Path("BACKEND_URL.txt").read_text().strip()

if not BACKEND_URL:
    BACKEND_URL = "http://127.0.0.1:8000"

headers = {"Authorization": f"Bearer {APP_API_TOKEN}"} if APP_API_TOKEN else {}

st.sidebar.success("Backend: " + BACKEND_URL)
st.title("CS 5588 – Week 7 App (Sudhakar Reddy Jerribanda)")
tabs = st.tabs(["Ask (QA)", "Generate (SD)", "Agent"])

with tabs[0]:
    st.subheader("Ask your project model")
    q = st.text_input("Question", "How do we visualize trust (BDI) in the app?")
    if st.button("Run QA"):
        try:
            r = requests.post(f"{BACKEND_URL.rstrip('/')}/qa", json={"query": q}, headers=headers, timeout=120)
            ct = (r.headers.get("content-type") or "").lower()
            if "application/json" in ct:
                data = r.json()
                st.markdown("**Answer:**")
                st.write(data.get("answer",""))
                st.markdown("**Citations:**")
                for c in data.get("citations", []):
                    st.write(f"- [{c.get('title','link')}]({c.get('url','#')})")
                st.caption(f"Latency: {data.get('latency_ms',0)} ms")
            else:
                st.warning("Non-JSON response from backend. Preview below:")
                st.code(r.text[:600])
        except Exception as e:
            st.error(f"Error: {e}")

with tabs[1]:
    st.subheader("Generate a project-style visual (Stable Diffusion)")
    prompt = st.text_area("Prompt", "infographic of a trust game with BDI panels, teal+indigo, flat vector, presentation-ready", height=100)
    if st.button("Generate Image"):
        try:
            r = requests.post(f"{BACKEND_URL.rstrip('/')}/generate", json={"prompt": prompt}, headers=headers, timeout=300)
            ct = (r.headers.get("content-type") or "").lower()
            if "application/json" in ct:
                data = r.json()
                fname = data.get("filename")
                if fname:
                    st.image(str(Path("Week_7")/"diffusion"/fname), caption=fname, use_column_width=True)
                st.caption(f"Latency: {data.get('latency_ms',0)} ms")
            else:
                st.warning("Non-JSON response from backend. Preview below:")
                st.code(r.text[:600])
        except Exception as e:
            st.error(f"Error: {e}")

with tabs[2]:
    st.subheader("Agent Orchestration (Plan → Execute → Aggregate)")
    ipt = st.text_area("Ask or request a visual", "Generate a clean isometric diagram of recruiter → MDT → ICT triage flow")
    if st.button("Run Agent"):
        try:
            r = requests.post(f"{BACKEND_URL.rstrip('/')}/agent", json={"input": ipt}, headers=headers, timeout=300)
            ct = (r.headers.get("content-type") or "").lower()
            if "application/json" in ct:
                data = r.json()
                st.markdown("**Final:**")
                st.write(data.get("final",""))
                st.markdown("**Hops (trace):**")
                for h in data.get("hops", []):
                    st.code(json.dumps(h, indent=2))
                img = data.get("image_filename")
                if img:
                    st.image(str(Path("Week_7")/"diffusion"/img), caption=img, use_column_width=True)
                if data.get("citations"):
                    st.markdown("**Citations:**")
                    for c in data["citations"]:
                        st.write(f"- [{c.get('title','link')}]({c.get('url','#')})")
                st.caption(f"Latency: {data.get('latency_ms',0)} ms")
            else:
                st.warning("Non-JSON response from backend. Preview below:")
                st.code(r.text[:600])
        except Exception as e:
            st.error(f"Error: {e}")
