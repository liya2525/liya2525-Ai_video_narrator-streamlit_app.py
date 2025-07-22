import streamlit as st
from PIL import Image
from moviepy.editor import VideoFileClip
from gtts import gTTS
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(page_title="🎥 AI Video Narrator", layout="centered")
st.title("🎬 AI Video Narrator")
st.markdown("Upload a short video; I’ll describe a key frame and narrate it.")

uploaded = st.file_uploader("📤 Upload a video", type=["mp4", "mov", "avi", "mkv"])
if uploaded:
    with open("input.mp4", "wb") as f: f.write(uploaded.read())
    st.video("input.mp4")

    clip = VideoFileClip("input.mp4")
    frame = clip.get_frame(clip.duration / 2)
    img = Image.fromarray(frame)
    img.save("frame.jpg")
    st.image(img, caption="📸 Key Frame from Video")

    st.info("🔍 Generating caption...")
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inp = proc(images=img, return_tensors="pt")
    out = model.generate(**inp)
    caption = proc.decode(out[0], skip_special_tokens=True)
    st.success(f"📝 Caption: {caption}")

    st.info("🔊 Narrating caption...")
    tts = gTTS(caption)
    tts.save("narration.mp3")
    st.audio("narration.mp3")
