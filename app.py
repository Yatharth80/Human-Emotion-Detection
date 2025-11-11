# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
from transformers import pipeline
import google.generativeai as genai
import tempfile
import os
from collections import Counter
import pandas as pd
from typing import Optional

# -----------------------------
# App title & layout
# -----------------------------
st.set_page_config(page_title="Emotion Recognition App", layout="wide")
st.title("😊 Emotion Recognition from Images, Video & Text")

# -----------------------------
# Compute target device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Model Benchmark Accuracy (Reference)
# -----------------------------
st.sidebar.markdown("### 📊 Model Benchmark Accuracy (Reference)")
st.sidebar.info("""
**Image Model (EmotiEffLib / FER-2013)**  
• Accuracy: ~72–80%

**Text Model (DistilBERT-GoEmotions)**  
• Accuracy: ~92% (micro-F1 score)

**Video Model**  
• Derived from image model — accuracy varies with lighting and clarity.
""")

# -----------------------------
# Initialize face emotion models
# -----------------------------
try:
    mtcnn = MTCNN(keep_all=True, post_process=False, min_face_size=40, device=device)
    model_name = get_model_list()[0]
    fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)
except Exception as e:
    st.error(f"❌ Face model initialization failed: {e}")
    st.stop()

# -----------------------------
# Initialize text emotion model (PyTorch-only)
# -----------------------------
@st.cache_resource
def load_text_emotion_model():
    return pipeline(
        "text-classification",
        model="joeddav/distilbert-base-uncased-go-emotions-student",
        framework="pt",
        return_all_scores=True
    )

text_emotion_analyzer = load_text_emotion_model()

# -----------------------------
# Gemini API Configuration
# -----------------------------
st.sidebar.header("🔑 Gemini API Configuration")
gemini_api_key = st.sidebar.text_input(
    "Enter your Gemini API Key",
    value="",
    type="password",
    help="Get your API key from https://makersuite.google.com/app/apikey"
)

# -----------------------------
# Helper function
# -----------------------------
def classify_emotion_type(emotion):
    negative = ['sad', 'anger', 'fear', 'disgust', 'contempt', 'anxious', 'worried']
    positive = ['happy', 'joy', 'excited', 'calm', 'peaceful', 'content']
    e = emotion.lower()
    if any(n in e for n in negative): return "negative"
    if any(p in e for p in positive): return "positive"
    return "neutral"


def softmax_confidence(logits) -> Optional[float]:
    try:
        arr = np.array(logits, dtype=np.float32).flatten()
        if arr.size == 0:
            return None
        arr = arr - np.max(arr)
        exp = np.exp(arr)
        denom = np.sum(exp)
        if denom == 0:
            return None
        probs = exp / denom
        return float(np.max(probs))
    except Exception:
        return None


def get_gemini_model(api_key: str):
    if not api_key:
        raise ValueError("Missing Gemini API key.")

    cache = st.session_state.setdefault("_gemini_model_cache", {})
    cached = cache.get("model")

    if not cached or cached.get("key") != api_key:
        genai.configure(api_key=api_key)
        cache["model"] = {
            "key": api_key,
            "instance": genai.GenerativeModel(model_name="gemini-1.5-flash")
        }

    return cache["model"]["instance"]


def get_emotion_guidance(emotion: str, emotion_type: str, context: str, api_key: str):
    try:
        model = get_gemini_model(api_key)
    except Exception as exc:
        return None, f"Gemini configuration error: {exc}"

    prompt = f"""
You are an empathetic emotional well-being coach.
Detected emotion: {emotion} (category: {emotion_type}).
Additional context:
{context if context else "No extra context provided."}

Provide:
1. A concise explanation of what someone feeling this emotion may be experiencing.
2. Three practical, supportive actions they can take right now.
3. One encouraging affirmation tailored to them.

Keep the tone warm, supportive, and grounded. Use short paragraphs and bullet lists when helpful.
"""

    try:
        response = model.generate_content(prompt.strip())
        if hasattr(response, "text"):
            return response.text, None
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts") and candidate.content.parts:
                return candidate.content.parts[0].text, None
        return str(response), None
    except Exception as exc:
        return None, f"Gemini response error: {exc}"

# ======================================================
# SECTION 1: Image-based Emotion Recognition
# ======================================================
st.header("🖼️ Emotion Recognition from Face Images")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)
        bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)

        if bounding_boxes is None:
            st.warning("😕 No face detected. Try another image.")
        else:
            faces = []
            annotated = frame.copy()

            for (x1, y1, x2, y2), p in zip(bounding_boxes, probs):
                if p and p > 0.9:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    face_img = frame[y1:y2, x1:x2]
                    emotion, logits = fer.predict_emotions(face_img, logits=True)

                    if isinstance(emotion, np.ndarray):
                        emotion_list = emotion.tolist()
                    elif isinstance(emotion, (list, tuple)):
                        emotion_list = list(emotion)
                    else:
                        emotion_list = [emotion]

                    top_emotion = emotion_list[0] if emotion_list else str(emotion)
                    confidence = softmax_confidence(logits) if logits is not None else None

                    faces.append({
                        'image': face_img,
                        'top_emotion': top_emotion,
                        'confidence': confidence,
                        'all_emotions': emotion_list,
                        'bbox': (x1, y1, x2, y2)
                    })

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{top_emotion}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            st.subheader("Results")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(annotated, caption="Detected faces with emotions", use_column_width=True)
            with col2:
                for i, f in enumerate(faces, 1):
                    st.image(f['image'], caption=f"Face {i}: {f['top_emotion']}", width=150)
                    if f['confidence'] is not None:
                        st.write(f"**Confidence:** {f['confidence']*100:.2f}%")

            conf_values = [f['confidence'] for f in faces if f['confidence'] is not None]
            if conf_values:
                avg_conf = float(np.mean(conf_values))
                st.metric("Estimated Accuracy", f"{avg_conf*100:.2f}%")
            else:
                st.metric("Estimated Accuracy", "N/A")

            if faces:
                st.divider()
                st.subheader("💡 Personalized Guidance")

                dominant_face = max(faces, key=lambda item: item['confidence'] or 0)
                dominant_emotion = dominant_face['top_emotion']
                emotion_type = classify_emotion_type(dominant_emotion)

                summary_lines = [
                    f"Face {idx+1}: {face['top_emotion']}" + (
                        f" ({face['confidence']*100:.1f}% confidence)" if face['confidence'] is not None else ""
                    )
                    for idx, face in enumerate(faces)
                ]
                guidance_context = "\n".join(summary_lines)

                st.write(f"Dominant emotion detected: **{dominant_emotion.capitalize()}** ({emotion_type}).")

                if not gemini_api_key:
                    st.info("Add your Gemini API key in the sidebar to receive personalized support and guidance.")
                else:
                    user_notes = st.text_area(
                        "Add optional personal context (e.g., situation, triggers, goals)",
                        key="image_guidance_context",
                        placeholder="e.g., Preparing for a big presentation tomorrow and feeling anxious."
                    )
                    if st.button("Generate Guidance", key="image_guidance_btn"):
                        with st.spinner("🧠 Crafting supportive guidance..."):
                            guidance, error = get_emotion_guidance(
                                dominant_emotion,
                                emotion_type,
                                guidance_context + ("\nUser notes: " + user_notes if user_notes else ""),
                                gemini_api_key
                            )
                        if error:
                            st.error(error)
                        elif guidance:
                            st.markdown(guidance)
                        else:
                            st.warning("No guidance generated. Try again in a moment.")

    except Exception as e:
        st.error(f"⚠️ Error during processing: {e}")

# ======================================================
# SECTION 2: Video-based Emotion Recognition
# ======================================================
st.header("🎬 Emotion Recognition from Video")

uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"], key="video")

if uploaded_video is not None:
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.info(f"Video: {total} frames, {fps} FPS")
        frame_skip = max(1, fps // 2)
        progress = st.progress(0)
        detected_emotions, confidences = [], []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, probs = mtcnn.detect(rgb)
                if boxes is not None:
                    for (x1, y1, x2, y2), p in zip(boxes, probs):
                        if p and p > 0.9:
                            face_img = rgb[int(y1):int(y2), int(x1):int(x2)]
                            try:
                                emotion, logits = fer.predict_emotions(face_img, logits=True)
                                if isinstance(emotion, (list, np.ndarray)):
                                    emotion = emotion[0]
                                detected_emotions.append(str(emotion))
                                if logits is not None:
                                    conf = softmax_confidence(logits)
                                    if conf is not None:
                                        confidences.append(conf)
                            except:
                                continue
            frame_idx += 1
            progress.progress(min(frame_idx/total, 1.0))

        cap.release()
        os.unlink(tfile.name)

        if detected_emotions:
            counter = Counter(detected_emotions)
            st.subheader("Detected Emotions Summary")
            for emo, count in counter.most_common():
                st.write(f"**{emo.capitalize()}**: {count} times")

            if confidences:
                avg_conf = float(np.mean(confidences))
                st.metric("Estimated Accuracy", f"{avg_conf*100:.2f}%")
            else:
                st.metric("Estimated Accuracy", "N/A")

            st.divider()
            st.subheader("💡 Personalized Guidance")
            dominant_emotion = counter.most_common(1)[0][0]
            emotion_type = classify_emotion_type(dominant_emotion)
            st.write(f"Dominant emotion in video: **{dominant_emotion.capitalize()}** ({emotion_type}).")

            if not gemini_api_key:
                st.info("Add your Gemini API key in the sidebar to receive personalized support and guidance.")
            else:
                video_notes = st.text_area(
                    "Add optional personal context about the video (e.g., scenario, audience, challenges)",
                    key="video_guidance_context",
                    placeholder="e.g., Recording a practice pitch and feeling uncertain about the delivery."
                )
                summary = "\n".join([f"{emo.capitalize()}: {count} frames" for emo, count in counter.most_common()])
                if confidences:
                    summary += f"\nAverage confidence: {np.mean(confidences)*100:.1f}%"
                if st.button("Generate Guidance", key="video_guidance_btn"):
                    with st.spinner("🧠 Crafting supportive guidance..."):
                        guidance, error = get_emotion_guidance(
                            dominant_emotion,
                            emotion_type,
                            summary + ("\nUser notes: " + video_notes if video_notes else ""),
                            gemini_api_key
                        )
                    if error:
                        st.error(error)
                    elif guidance:
                        st.markdown(guidance)
                    else:
                        st.warning("No guidance generated. Try again in a moment.")
        else:
            st.warning("No emotions detected in video.")

    except Exception as e:
        st.error(f"⚠️ Video processing error: {e}")

# ======================================================
# SECTION 3: Text-based Emotion Recognition
# ======================================================
st.header("✍️ Emotion Recognition from Text")

user_text = st.text_area("Enter text (e.g., 'I just got promoted!' or 'I feel so lonely...')")

if user_text.strip():
    try:
        results = text_emotion_analyzer(user_text)[0]
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        top_emotion = results[0]
        st.success(f"Predicted Emotion: **{top_emotion['label']}**")
        st.write(f"Confidence Score: {top_emotion['score']*100:.2f}%")
        with st.expander("See detailed probabilities"):
            for r in results[:5]:
                st.write(f"{r['label']}: {r['score']*100:.2f}%")

        st.metric("Estimated Accuracy", "≈92% (Pretrained GoEmotions Model)")

        st.subheader("💡 Personalized Guidance")
        emotion_type = classify_emotion_type(top_emotion['label'])
        if not gemini_api_key:
            st.info("Add your Gemini API key in the sidebar to receive personalized support and guidance.")
        else:
            text_notes = st.text_area(
                "Add optional personal context to tailor the guidance",
                key="text_guidance_context",
                placeholder="e.g., I'm feeling this way because of a recent conflict with a friend."
            )
            if st.button("Generate Guidance", key="text_guidance_btn"):
                with st.spinner("🧠 Crafting supportive guidance..."):
                    guidance, error = get_emotion_guidance(
                        top_emotion['label'],
                        emotion_type,
                        f"Model probability distribution: {[(r['label'], round(r['score']*100, 1)) for r in results[:5]]}"
                        + ("\nUser notes: " + text_notes if text_notes else ""),
                        gemini_api_key
                    )
                if error:
                    st.error(error)
                elif guidance:
                    st.markdown(guidance)
                else:
                    st.warning("No guidance generated. Try again in a moment.")

    except Exception as e:
        st.error(f"⚠️ Text analysis error: {e}")
