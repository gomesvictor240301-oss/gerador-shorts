"""
main.py — Backend do gerador de Shorts (FastAPI + yt-dlp + Whisper (Groq) + FFmpeg)

Fluxo:
  POST /generate-short  -> cria um "job" e devolve job_id na hora
  GET  /status/{job_id} -> o site pergunta de 2 em 2s: etapa atual / URL final
  GET  /outputs/xxx.mp4 -> arquivo pronto (pasta pública)

Por que "job + polling" e não esperar a resposta?
Renderizar leva 1-3 min. Hospedagens derrubam requisições longas (timeout).
Então respondemos rápido e o frontend consulta o andamento.
"""
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests                     # chama a API de transcrição (Groq)
import yt_dlp                       # baixa vídeos do YouTube
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ----------------------------- CONFIGURAÇÃO -----------------------------
BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "work"        # arquivos temporários (apagados no fim)
OUT_DIR = BASE_DIR / "outputs"      # pasta PÚBLICA com os .mp4 finais
WORK_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")             # chave grátis da Groq
GROQ_MODEL = "whisper-large-v3-turbo"                # modelo Whisper na nuvem
CLIP_SECONDS = int(os.getenv("CLIP_SECONDS", "60"))  # duração do corte
COOKIES_FILE = os.getenv("YT_COOKIES")               # opcional (veja README)
FILE_TTL = 60 * 60                                   # apaga vídeos após 1h

app = FastAPI(title="Gerador de Shorts")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=OUT_DIR), name="outputs")

# "Banco de dados" em memória: {job_id: {"status":..., "step":..., ...}}
# (para produção séria, troque por Redis/SQLite)
JOBS: dict[str, dict] = {}

# Processamos 1 vídeo por vez (servidor grátis tem pouca memória).
_render_lock = threading.Semaphore(1)


# ------------------------------- MODELOS --------------------------------
class GenerateRequest(BaseModel):
    url: str


YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com",
                 "youtu.be", "music.youtube.com"}


def validate_youtube_url(url: str) -> str:
    """Segurança: só aceitamos links do YouTube (evita abusos)."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or parsed.hostname not in YOUTUBE_HOSTS:
        raise HTTPException(400, "Cole um link válido do YouTube.")
    return url.strip()


# --------------------------- FUNÇÕES DO PIPELINE ------------------------
def set_step(job_id: str, step: str):
    JOBS[job_id]["step"] = step     # o frontend mostra esse texto


def download_video(url: str, folder: Path) -> Path:
    """PASSO A — baixa só os primeiros CLIP_SECONDS segundos em até 720p."""
    opts = {
        # melhor vídeo até 720p + melhor áudio, juntados em mp4
        "format": "bv*[height<=720]+ba/b[height<=720]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(folder / "input.%(ext)s"),
        # baixa apenas o trecho 0 -> CLIP_SECONDS (muito mais rápido!)
        "download_ranges": yt_dlp.utils.download_range_func(None, [(0, CLIP_SECONDS)]),
        "force_keyframes_at_cuts": True,
        "noplaylist": True,         # ignora playlists
        "quiet": True,
    }
    if COOKIES_FILE and os.path.exists(COOKIES_FILE):
        opts["cookiefile"] = COOKIES_FILE   # ajuda contra bloqueio anti-bot

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    video = next(folder.glob("input.*"), None)
    if not video:
        raise RuntimeError("Falha ao baixar o vídeo.")
    return video


def extract_audio(video: Path, folder: Path) -> Path:
    """Extrai áudio mono 16 kHz (formato ideal para o Whisper)."""
    audio = folder / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-t", str(CLIP_SECONDS),
         "-vn", "-ac", "1", "-ar", "16000", str(audio)],
        check=True, capture_output=True,
    )
    return audio


def transcribe_words(audio: Path) -> list[dict]:
    """PASSO B — envia o áudio à Groq (Whisper na nuvem) e recebe
    [{word, start, end}, ...]. Assim o servidor não precisa de GPU/RAM."""
    if not GROQ_API_KEY:
        raise RuntimeError("Falta a variável GROQ_API_KEY no servidor.")
    with open(audio, "rb") as f:
        resp = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("audio.wav", f, "audio/wav")},
            data={
                "model": GROQ_MODEL,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",   # tempo de cada palavra
            },
            timeout=120,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Erro na transcrição ({resp.status_code}): {resp.text[:200]}")
    words = []
    for w in resp.json().get("words", []):
        text = w["word"].strip()
        if text:
            words.append({"word": text, "start": w["start"], "end": w["end"]})
    return words


def ass_time(seconds: float) -> str:
    """Converte segundos para o formato de tempo do .ass (h:mm:ss.cc)."""
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def build_ass(words: list[dict], path: Path):
    """
    Cria o arquivo de legenda .ass com UMA palavra por vez.
    ASS permite estilo avançado: fonte grossa, contorno, efeito "pop".
    Cores no ASS são BGR: &H0000FFFF = amarelo.
    """
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        # Alignment 5 = centro da tela | Outline 7 = contorno preto grosso
        "Style: Viral,DejaVu Sans,96,&H0000FFFF,&H000000FF,&H00000000,&H00000000,"
        "1,0,0,0,100,100,0,0,1,7,2,5,60,60,0,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    lines = []
    for i, w in enumerate(words):
        start = w["start"]
        end = w["end"]
        # estende até a próxima palavra (evita "piscar" entre palavras)
        if i + 1 < len(words) and words[i + 1]["start"] - end < 0.3:
            end = words[i + 1]["start"]
        text = re.sub(r"[{}\\]", "", w["word"]).upper()   # CAIXA ALTA
        # \t = animação "pop": começa em 80% e cresce até 100% em 0,08s
        pop = r"{\fscx80\fscy80\t(0,80,\fscx100\fscy100)}"
        lines.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Viral,,0,0,0,,{pop}{text}"
        )
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def render_short(video: Path, ass_file: Path, output: Path):
    """
    PASSO C — FFmpeg: corta 60s, converte 16:9 -> 9:16 e queima legendas.
      crop=...   : recorta o centro do vídeo na proporção 9:16
      scale      : redimensiona para 1080x1920 (full HD vertical)
      ass=...    : "queima" as legendas na imagem
    """
    vf = ("crop=trunc(ih*9/16/2)*2:ih,"      # largura = altura*9/16 (par)
          "scale=1080:1920,setsar=1,"
          f"ass={ass_file.name}")             # roda dentro da pasta do job
    subprocess.run(
        ["ffmpeg", "-y", "-i", video.name, "-t", str(CLIP_SECONDS),
         "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
         str(output.resolve())],
        check=True, capture_output=True, cwd=video.parent,
    )


def cleanup_old_files():
    """Apaga vídeos antigos para não lotar o disco."""
    now = time.time()
    for f in OUT_DIR.glob("*.mp4"):
        if now - f.stat().st_mtime > FILE_TTL:
            f.unlink(missing_ok=True)


# ------------------------------ O JOB COMPLETO --------------------------
def process_job(job_id: str, url: str):
    folder = WORK_DIR / job_id
    folder.mkdir(exist_ok=True)
    try:
        set_step(job_id, "Na fila...")
        with _render_lock:                      # 1 vídeo por vez
            set_step(job_id, "Baixando vídeo...")
            video = download_video(url, folder)

            set_step(job_id, "Transcrevendo áudio...")
            audio = extract_audio(video, folder)
            words = transcribe_words(audio)
            if not words:
                raise RuntimeError("Não detectei fala nos primeiros 60s.")

            set_step(job_id, "Renderizando Short com Legendas...")
            ass_file = folder / "subs.ass"
            build_ass(words, ass_file)
            output = OUT_DIR / f"{job_id}.mp4"   # PASSO D: pasta pública
            render_short(video, ass_file, output)

        JOBS[job_id].update(status="done", step="Pronto!",
                            url=f"/outputs/{job_id}.mp4")
    except Exception as e:                       # qualquer erro vira mensagem
        JOBS[job_id].update(status="error", step="Erro",
                            error=f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        shutil.rmtree(folder, ignore_errors=True)  # limpa temporários
        cleanup_old_files()


# -------------------------------- ROTAS ---------------------------------
@app.post("/generate-short")
def generate_short(req: GenerateRequest, background: BackgroundTasks):
    url = validate_youtube_url(req.url)
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "processing", "step": "Iniciando..."}
    background.add_task(process_job, job_id, url)   # roda em segundo plano
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job não encontrado.")
    return job


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")   # serve o frontend
