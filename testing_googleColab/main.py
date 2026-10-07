from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
import requests
import json

# 1. Bina aplikasi FastAPI
app = FastAPI(title="EcoSustain RAG API")

# Keselamatan Pasukan: Benarkan Frontend (React/Flutter/HTML) untuk akses API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Dalam hackathon, kita benarkan semua akses supaya cepat
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Struktur data untuk menerima soalan daripada Frontend
class ChatRequest(BaseModel):
    question: str

# 3. Fungsi membaca fail PDF (Dijalankan sekali sahaja semasa API dihidupkan)
try:
    reader = PdfReader("dokumen_sdg.pdf")
    teks_pdf = ""
    for page in reader.pages:
        teks_pdf += page.extract_text()
    teks_pdf = teks_pdf[:2000] # Hadkan 2000 aksara untuk keselamatan memori lokal
    print("✅ Fail PDF berjaya dibaca dan sedia di dalam memori!")
except Exception as e:
    teks_pdf = "Tiada dokumen ditemui."
    print("❌ Gagal membaca fail PDF. Pastikan 'dokumen_sdg.pdf' ada dalam folder.")

# 4. Pautan Jalan (Endpoint API) untuk Frontend memanggil AI
@app.post("/chat")
def tanya_ai(data: ChatRequest):
    soalan_anda = data.question
    
    # Rangka Prompt RAG
    prompt_rag = f"""
    Anda adalah AI pembantu kelestarian pasukan hackathon. Jawab soalan pengguna berdasarkan KONTEKS dokumen yang diberikan sahaja. 
    Jika jawapan tiada dalam konteks dokumen, katakan anda tidak tahu.

    KONTEKS DOKUMEN:
    {teks_pdf}

    SOALAN: {soalan_anda}
    JAWAPAN:
    """

    # Hantar ke Ollama lokal (Qwen)
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen",
        "prompt": prompt_rag,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        result = json.loads(response.text)
        # Pulangkan jawapan kepada Frontend dalam format standard JSON
        return {"status": "success", "reply": result["response"]}
    except Exception as e:
        return {"status": "error", "message": f"Gagal bersambung ke Ollama: {str(e)}"}

# Endpoint ringkas untuk semak status API
@app.get("/")
def semak_status():
    return {"status": "Sistem API EcoSustain sedang berjalan dengan lancar!"}
