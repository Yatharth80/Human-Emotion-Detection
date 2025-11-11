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
import time
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
if device == "cuda":
    st.success(f"🚀 Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    st.warning("⚠ Running on CPU")

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
# Helper functions
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

@st.cache_data(show_spinner=False)
def get_available_models(api_key):
    """Get list of available Gemini models"""
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        available = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available.append(model.name.replace('models/', ''))
        return available
    except:
        return []

def get_emotion_guidance(emotion: str, emotion_type: str, context: str, api_key: str):
    """Get personalized advice from Gemini based on detected emotion"""
    if not api_key:
        return None, "Please enter your Gemini API key in the sidebar."
    
    try:
        genai.configure(api_key=api_key)
        
        # Try to find a working model - check available models first
        model_names_to_try = []
        
        try:
            # Try to list available models
            available_models = get_available_models(api_key)
            if available_models:
                model_names_to_try = available_models[:3]  # Try first 3 available
        except:
            pass
        
        # Fallback to standard model names if we couldn't list them
        if not model_names_to_try:
            model_names_to_try = [
                'gemini-pro',  # Most widely available
                'models/gemini-pro',  # Try with models/ prefix
                'gemini-1.0-pro',
                'models/gemini-1.0-pro',
                'gemini-1.5-flash',
                'models/gemini-1.5-flash',
                'gemini-1.5-pro',
                'models/gemini-1.5-pro'
            ]
        
        # Prepare the prompt first (same for all models)
        if emotion_type == "negative":
            prompt = f"""The emotion detected is: {emotion}

This appears to be a negative emotion. Please provide:
1. A brief, empathetic explanation of why this emotion might occur
2. 3-5 practical, actionable steps to help overcome or manage this emotion
3. Some encouragement or positive perspective

Keep the response concise (2-3 paragraphs), warm, and supportive. Focus on actionable advice."""
        elif emotion_type == "positive":
            prompt = f"""The emotion detected is: {emotion}

This appears to be a positive emotion! Please provide:
1. A brief explanation of the benefits of this positive state
2. 3-5 ways to maintain, amplify, or cultivate this positive emotion
3. How to create more moments like this in daily life

Keep the response concise (2-3 paragraphs), uplifting, and practical."""
        else:
            prompt = f"""The emotion detected is: {emotion}

Additional context: {context if context else "No extra context provided."}

Please provide:
1. A brief explanation of this emotional state
2. Some insights about this emotion
3. Suggestions for emotional awareness and regulation

Keep the response concise (2-3 paragraphs) and balanced."""
        
        # Try each model with the actual prompt
        last_error = None
        for model_name in model_names_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt.strip())
                # If we got here, the model works - extract and return
                if hasattr(response, 'text'):
                    return response.text, None
                elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                    if hasattr(response.candidates[0], 'content'):
                        if hasattr(response.candidates[0].content, 'parts'):
                            return response.candidates[0].content.parts[0].text, None
                return str(response), None
            except Exception as e:
                last_error = str(e)
                continue
        
        # If we get here, all models failed
        return None, f"Could not find a working Gemini model. Please check your API key at https://makersuite.google.com/app/apikey and ensure you have access to Gemini models. Last error: {last_error}"
        
    except Exception as e:
        error_msg = str(e)
        return None, f"Error generating advice: {error_msg}"

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

            if faces and gemini_api_key:
                st.divider()
                st.subheader("💡 Personalized Guidance")
                dominant_face = max(faces, key=lambda item: item['confidence'] or 0)
                dominant_emotion = dominant_face['top_emotion']
                emotion_type = classify_emotion_type(dominant_emotion)
                with st.spinner("🤖 Getting personalized guidance from Gemini..."):
                    context = "\n".join([f"Face {i+1}: {f['top_emotion']} ({(f['confidence'] or 0)*100:.1f}%)" for i, f in enumerate(faces)])
                    guidance, error = get_emotion_guidance(dominant_emotion, emotion_type, context, gemini_api_key)
                if error:
                    st.error(error)
                elif guidance:
                    st.markdown("### 📝 Personalized Guidance")
                    st.markdown(guidance)

    except Exception as e:
        st.error(f"⚠️ Error during processing: {e}")

# ======================================================
# SECTION 2: Video-based Emotion Recognition
# ======================================================
st.header("🎬 Emotion Recognition from Video")

uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"], key="video")

if uploaded_video is not None:
    tfile_path = None
    cap = None
    try:
        # Create temporary file and write video data
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_video.read())
            tfile_path = tfile.name
        
        # Open video with OpenCV
        cap = cv2.VideoCapture(tfile_path)
        if not cap.isOpened():
            st.error("❌ Could not open video file. Please try another format.")
        else:
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
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
                                    conf = softmax_confidence(logits)
                                    if conf is not None:
                                        confidences.append(conf)
                                except:
                                    continue
                frame_idx += 1
                progress.progress(min(frame_idx/total, 1.0) if total > 0 else 0)

            # Close video capture before deleting file
            if cap:
                cap.release()
                cap = None
            
            progress.empty()

            if detected_emotions:
                counter = Counter(detected_emotions)
                st.subheader("Detected Emotions Summary")
                for emo, count in counter.most_common():
                    st.write(f"**{emo.capitalize()}**: {count} times")
                avg_conf = np.mean(confidences) if confidences else 0
                st.metric("Estimated Accuracy", f"{avg_conf*100:.2f}%" if avg_conf else "N/A")

                if gemini_api_key:
                    st.divider()
                    st.subheader("💡 Personalized Guidance")
                    dominant_emotion = counter.most_common(1)[0][0]
                    emotion_type = classify_emotion_type(dominant_emotion)
                    with st.spinner("🤖 Getting personalized guidance from Gemini..."):
                        summary = "\n".join([f"{emo.capitalize()}: {count} frames" for emo, count in counter.most_common()])
                        if confidences:
                            summary += f"\nAverage confidence: {np.mean(confidences)*100:.1f}%"
                        guidance, error = get_emotion_guidance(dominant_emotion, emotion_type, summary, gemini_api_key)
                    if error:
                        st.error(error)
                    elif guidance:
                        st.markdown("### 📝 Personalized Guidance")
                        st.markdown(guidance)
            else:
                st.warning("😕 No emotions detected in the video.")

    except Exception as e:
        st.error(f"⚠️ Video processing error: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        # Cleanup: Release video capture and delete temp file
        if cap:
            cap.release()
        if tfile_path and os.path.exists(tfile_path):
            try:
                # Small delay to ensure file handle is released (Windows-specific)
                time.sleep(0.1)
                os.unlink(tfile_path)
            except Exception:
                # If deletion fails, it's okay - temp files will be cleaned up eventually
                pass

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
        st.metric("Estimated Accuracy", "≈92% (Pretrained GoEmotions Model)")
        with st.expander("See detailed probabilities"):
            for r in results[:5]:
                st.write(f"{r['label']}: {r['score']*100:.2f}%")

        if gemini_api_key:
            st.divider()
            st.subheader("💡 Personalized Guidance")
            emotion_type = classify_emotion_type(top_emotion['label'])
            with st.spinner("🤖 Getting personalized guidance from Gemini..."):
                context_summary = f"Text emotion: {top_emotion['label']} ({top_emotion['score']*100:.2f}%)"
                guidance, error = get_emotion_guidance(top_emotion['label'], emotion_type, context_summary, gemini_api_key)
            if error:
                st.error(error)
            elif guidance:
                st.markdown("### 📝 Personalized Guidance")
                st.markdown(guidance)

    except Exception as e:
        st.error(f"⚠️ Text analysis error: {e}")
