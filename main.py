# test.py — Flask chatbot + AI Image Generator + GIF Generator
from flask import Flask, request, jsonify, render_template_string, send_file, Response
import google.generativeai as genai
import pyttsx3
import os, re, io
from datetime import datetime
from PIL import Image

from diffusers import StableDiffusionPipeline
import torch

# -------------------- CONFIG --------------------
API_KEY = "YOUR_API_KEY"  # Gemini API key
genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-1.5-flash"

app = Flask(__name__)
os.makedirs("static/audio", exist_ok=True)
os.makedirs("static/images", exist_ok=True)

# In-memory chat history
chat_history = []

# -------------------- Helpers --------------------
def clean_text(txt: str) -> str:
    txt = re.sub(r"\*", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def text_to_speech(text: str) -> str:
    text = text[:5000]
    filename = f"static/audio/{int(datetime.utcnow().timestamp())}.mp3"
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename

def ask_gemini(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(prompt)
        return clean_text(resp.text or "")
    except Exception as e:
        return f"⚠️ Error fetching response: {e}"

# -------------------- Stable Diffusion --------------------
pipe = None
def load_sd_model():
    global pipe
    if pipe is None:
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
    return pipe

# -------------------- GIF Generator (from gifimg.py) --------------------
def generate_sd_image(prompt: str, size=(512, 512)):
    pipe = load_sd_model()
    img = pipe(prompt).images[0]
    img = img.resize(size)
    return img

def make_gif(img):
    frames = []
    width, height = img.size
    for i in range(8):
        scale = 1 + i * 0.07
        rotated = img.rotate(i * 3)
        frame = rotated.resize((int(width * scale), int(height * scale)))
        frame = frame.crop((0, 0, width, height))
        frames.append(frame)

    gif_bytes = io.BytesIO()
    frames[0].save(
        gif_bytes,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0
    )
    gif_bytes.seek(0)
    return gif_bytes

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template_string(""" 
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Gemini Chatbot + AI Image + GIF Generator</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
:root { --bg:#0f172a; --card:#111827; --accent:#2563eb; --bot:#1f2937; --user:#0ea5e9; }
body { margin:0; font-family:ui-sans-serif,system-ui; background: var(--bg); color:#e5e7eb; }
.container { max-width: 900px; margin: 32px auto; padding: 0 16px; }
.title { font-size:28px; font-weight:700; margin-bottom:16px; }
.card { background: var(--card); border-radius:16px; padding:16px; box-shadow:0 10px 30px rgba(0,0,0,.35); }
#messages { height: 40vh; overflow-y:auto; padding:8px; display:flex; flex-direction:column; gap:10px; }
.msg { max-width:80%; padding:10px 12px; border-radius:14px; line-height:1.4; white-space:pre-wrap; }
.msg.bot { background: var(--bot); align-self:flex-start; }
.msg.user { background: var(--user); align-self:flex-end; }
.row { display:flex; gap:8px; margin-top:12px; }
.input { flex:1; background:#0b1220; border:1px solid #1f2937; color:#e5e7eb; border-radius:12px; padding:12px; outline:none; }
.btn { background: var(--accent); border:none; color:white; padding:12px 14px; border-radius:12px; cursor:pointer; font-weight:600; }
.btn.secondary { background:#374151; }
.btn:disabled { opacity:.6; cursor:not-allowed; }
audio { width:100%; margin-top:8px; display:none; }
#image-generator, #gif-generator { display:none; margin-top:20px; }
#sd-image, #gif-preview { max-width:100%; margin-top:10px; border-radius:12px; }
</style>
</head>
<body>
<div class="container">
  <div class="title">🤖 Gemini Chatbot + AI Image + GIF Generator</div>
  <div class="card">
    <div id="messages"></div>
    <div class="row">
      <input id="textInput" class="input" placeholder="Type your message..." />
      <div class="row" style="gap:4px;">
        <button id="sendBtn" class="btn">Send</button>
        <button id="voiceBtn" class="btn secondary">🎤 Voice</button>
        <button id="openImgBtn" class="btn secondary">🖼️ Image</button>
        <button id="openGifBtn" class="btn secondary">🎞️ GIF</button>
      </div>
    </div>
    <audio id="player" controls></audio>

    <!-- Image generator -->
    <div id="image-generator">
      <input id="promptInput" class="input" placeholder="Enter image prompt..." style="margin-top:10px;"/>
      <button id="genBtn" class="btn" style="margin-top:8px;">Generate Image</button>
      <img id="sd-image" src="" />
    </div>

    <!-- GIF generator -->
    <div id="gif-generator">
      <input id="gifPrompt" class="input" placeholder="Enter GIF prompt..." style="margin-top:10px;"/>
      <select id="gifSize" class="input" style="margin-top:10px;">
        <option value="256">256x256</option>
        <option value="512" selected>512x512</option>
        <option value="768">768x768</option>
      </select>
      <button id="gifBtn" class="btn" style="margin-top:8px;">Generate GIF</button>
      <img id="gif-preview" src="" />
    </div>
  </div>
</div>

<script>
const messagesEl=document.getElementById("messages");
const inputEl=document.getElementById("textInput");
const sendBtn=document.getElementById("sendBtn");
const voiceBtn=document.getElementById("voiceBtn");
const openImgBtn=document.getElementById("openImgBtn");
const openGifBtn=document.getElementById("openGifBtn");
const player=document.getElementById("player");
const imgGenDiv=document.getElementById("image-generator");
const gifGenDiv=document.getElementById("gif-generator");
const promptInput=document.getElementById("promptInput");
const genBtn=document.getElementById("genBtn");
const sdImage=document.getElementById("sd-image");
const gifPrompt=document.getElementById("gifPrompt");
const gifSize=document.getElementById("gifSize");
const gifBtn=document.getElementById("gifBtn");
const gifPreview=document.getElementById("gif-preview");

function addMessage(role,text,speakable=false){
  const div=document.createElement("div");
  div.className="msg "+role;
  div.textContent=text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop=messagesEl.scrollHeight;
}

async function sendMessage(text){
  if(!text)return;
  addMessage("user",text);
  inputEl.value="";
  sendBtn.disabled=true;
  try{
    const res=await fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:text})});
    const data=await res.json();
    addMessage("bot",data.reply,true);
  }catch(e){addMessage("bot","Error contacting server.");}
  finally{sendBtn.disabled=false;}
}

sendBtn.onclick=()=>sendMessage(inputEl.value.trim());
inputEl.addEventListener("keydown",(e)=>{if(e.key==="Enter")sendMessage(inputEl.value.trim());});

// Show image/gif generator
openImgBtn.onclick=()=>{ imgGenDiv.style.display=imgGenDiv.style.display==="none"?"block":"none"; }
openGifBtn.onclick=()=>{ gifGenDiv.style.display=gifGenDiv.style.display==="none"?"block":"none"; }

// Generate AI image
genBtn.onclick=async()=>{
  const prompt=promptInput.value.trim();
  if(!prompt)return;
  genBtn.disabled=true;
  genBtn.textContent="Generating...";
  try{
    const res=await fetch("/generate_image",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt})});
    const data=await res.json();
    if(data.url){ sdImage.src=data.url; }
  }finally{ genBtn.disabled=false; genBtn.textContent="Generate Image"; }
}

// Generate GIF
gifBtn.onclick=async()=>{
  const prompt=gifPrompt.value.trim();
  const size=gifSize.value;
  if(!prompt)return;
  gifBtn.disabled=true;
  gifBtn.textContent="Generating...";
  try{
    const formData=new FormData();
    formData.append("prompt",prompt);
    formData.append("size",size);
    const res=await fetch("/generate_gif",{method:"POST",body:formData});
    const blob=await res.blob();
    const url=URL.createObjectURL(blob);
    gifPreview.src=url;
  }finally{ gifBtn.disabled=false; gifBtn.textContent="Generate GIF"; }
}
</script>
</body>
</html>
    """)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()
    if not user_text: return jsonify({"reply":"Please type something."})
    chat_history.append({"role":"user","text":user_text,"ts":datetime.utcnow().isoformat()})
    bot_reply = ask_gemini(user_text)
    chat_history.append({"role":"bot","text":bot_reply,"ts":datetime.utcnow().isoformat()})
    return jsonify({"reply":bot_reply})

@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json(silent=True) or {}
    text = clean_text(data.get("text") or "")
    if not text: return jsonify({"error":"No text provided"}),400
    tts_path = text_to_speech(text)
    return jsonify({"audio_url": f"/{tts_path}"})

@app.route("/generate_image", methods=["POST"])
def generate_image():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt","").strip()
    if not prompt: return jsonify({"error":"No prompt"}),400
    pipe = load_sd_model()
    image = pipe(prompt).images[0]
    filename = f"static/images/{int(datetime.utcnow().timestamp())}.png"
    image.save(filename)
    return jsonify({"url": f"/{filename}"})

@app.route("/generate_gif", methods=["POST"])
def generate_gif():
    prompt = request.form.get("prompt", "A cute Pikachu")
    size_val = int(request.form.get("size", "512"))
    img = generate_sd_image(prompt, size=(size_val, size_val))
    gif_bytes = make_gif(img)
    return send_file(gif_bytes, mimetype="image/gif")

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
