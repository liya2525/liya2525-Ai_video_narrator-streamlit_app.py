import streamlit as st
import os, shutil, base64
from PIL import Image
import torch
from transformers import AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
from auto_gptq import AutoGPTQForCausalLM
from moviepy.editor import VideoFileClip, AudioFileClip
from gtts import gTTS
import tempfile
import re

st.set_page_config(page_title="🎙️ AI Video Narrator", layout="centered")
st.title("🎥 AI Video Narrator")
st.markdown("Upload a short video. The app will analyze it and narrate what it sees.")

# Cleanup and folders
shutil.rmtree("frames", ignore_errors=True)
os.makedirs("frames", exist_ok=True)

# Upload video
uploaded_file = st.file_uploader("📂 Upload Video", type=["mp4", "mov", "avi"])
if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)
    st.info("📸 Extracting frames...")
    clip = VideoFileClip(video_path)
    duration = int(clip.duration)
    for i in range(duration):
        frame = clip.get_frame(i)
        Image.fromarray(frame).save(f"frames/frame_{i:03d}.jpg")

    st.success(f"Extracted {duration} frames")

    st.info("🤖 Loading BLIP captioning model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    def get_caption(image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = blip_model.generate(**inputs)
        return processor.decode(output[0], skip_special_tokens=True)

    st.info("📝 Generating captions for frames...")
    captions = []
    for fname in sorted(os.listdir("frames")):
        cap = get_caption(f"frames/{fname}")
        captions.append(cap)
    st.success("Captions generated!")

    st.info("🧠 Summarizing with Vicuna...")
    model_name = "TheBloke/vicuna-7B-v1.5-GPTQ"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    vicuna_model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        model_basename="model",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    prompt = (
        "You are an AI narrator trained in the style of AI description. "
        "Given the following frame-level captions extracted from a video, "
        "generate a descriptive and immersive narration. Focus on accurate, elegant language and a calm, observational tone. "
        "Avoid storytelling or dialogue. Just describe what is seen, as if narrating a nature or documentary scene.\n\n"
        + "\n".join(f"- {c}" for c in captions) +
        "\n\nNarrate this sequence in a BBC Earth style:"
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output = vicuna_model.generate(
            input_ids=input_ids,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    narration = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()

    def clean_cutoff(text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        return ' '.join(sentences[:-1]) if not text.strip().endswith(('.', '!', '?')) else text.strip()

    narration = clean_cutoff(narration)
    st.success("🗣️ Narration generated!")

    st.text_area("Narration Text", narration, height=200)

    st.info("🔊 Generating audio...")
    tts = gTTS(narration)
    tts.save("narration.mp3")

    st.audio("narration.mp3")

    st.info("🎬 Merging audio and video...")
    final_clip = clip.set_audio(AudioFileClip("narration.mp3"))
    final_clip.write_videofile("final_video.mp4", codec="libx264", audio_codec="aac")
    with open("final_video.mp4", "rb") as f:
        video_bytes = f.read()
    st.video(video_bytes)
    st.download_button("⬇️ Download Narrated Video", data=video_bytes, file_name="narrated_video.mp4")
