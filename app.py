import streamlit as st
import tempfile
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Single Video Player Re-ID", layout="wide")
st.title("⚽ Single Video: Player Re-Identification")

# Load ResNet18 for feature embedding
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = models.resnet18(pretrained=True).to(device).eval()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(img_tensor)
    return features.squeeze().cpu().numpy()

def reidentify_players(video_path, model, conf_thresh=0.3, sim_thresh=0.85, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    player_db = []
    next_id = 0
    annotated_frames = []
    processed = 0

    while processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("❌ Skipped a corrupt or missing frame.")
            continue

        results = model(frame)
        detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            if class_name.lower() in ["player", "person"] and conf > conf_thresh:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                feat = extract_features(crop)
                detections.append({'bbox': (x1, y1, x2, y2), 'feature': feat})

        for det in detections:
            matched = False
            for p in player_db:
                sim = cosine_similarity([det['feature']], [p['feature']])[0][0]
                if sim > sim_thresh:
                    det['id'] = p['id']
                    p['feature'] = det['feature']  # Update DB feature
                    matched = True
                    break
            if not matched:
                det['id'] = next_id
                player_db.append({'id': next_id, 'feature': det['feature']})
                next_id += 1

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {det['id']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        annotated_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        processed += 1

    cap.release()
    return annotated_frames

# Sidebar inputs
with st.sidebar:
    uploaded_model = st.file_uploader("📦 Upload YOLOv8 Model (best.pt)", type=["pt"])
    uploaded_video = st.file_uploader("🎥 Upload Soccer Video", type=["mp4"])
    max_frames = st.slider("🎞️ Max Frames to Process", 30, 500, 150, step=10)
    run = st.button("🚀 Run Re-Identification")

# Process the video if inputs are provided
if run and uploaded_model and uploaded_video:
    with st.spinner("🔍 Processing video and identifying players..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
            tmp_model.write(uploaded_model.read())
            model_path = tmp_model.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_video.read())
            video_path = tmp_video.name

        model = YOLO(model_path)

        # Store processed frames in session state
        st.session_state.output_frames = reidentify_players(
            video_path, model, max_frames=max_frames)

        st.success(f"✅ Processed {len(st.session_state.output_frames)} frames!")

# Frame viewing section
if "output_frames" in st.session_state:
    frames = st.session_state.output_frames
    st.subheader("🖼️ View Annotated Frames")
    frame_idx = st.slider("Select Frame", 0, len(frames) - 1, 0)
    st.image(frames[frame_idx], caption=f"Frame {frame_idx}", use_column_width=True)
else:
    st.info("Upload a model and a soccer video, then click 'Run Re-Identification' to begin.")
