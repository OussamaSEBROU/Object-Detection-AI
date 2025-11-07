import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import time
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Real-Time YOLOv8 Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title ---
st.title("🤖 Real-Time Object Detection with YOLOv8")
st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05,
                               help="Minimum probability to filter weak detections.")

# --- Model Loading ---
@st.cache_resource
def load_model(model_path):
    """
    Loads the YOLO model from the specified path.
    Caches the model to avoid reloading on every app rerun.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_path = 'yolov8n.pt'
model = load_model(model_path)

if model:
    st.sidebar.success(f"Model '{os.path.basename(model_path)}' loaded successfully.")
else:
    st.sidebar.error("Model failed to load. Please check the path or file.")
    st.stop()  # Stop the app if model fails to load

# --- Session State Initialization ---
if 'stop' not in st.session_state:
    st.session_state.stop = True
if 'video_file_path' not in st.session_state:
    st.session_state.video_file_path = None
if 'save_video' not in st.session_state:
    st.session_state.save_video = False

# --- Input Source Selection ---
st.sidebar.markdown("---")
st.sidebar.header("Input Source")
source_option = st.sidebar.radio("Choose input source",
                                 ("Upload a file (Image/Video)", "Live Camera"),
                                 index=0, key="source_option")

# --- Main Logic ---
frame_placeholder = st.empty()

if source_option == "Live Camera":
    st.sidebar.markdown("---")
    st.sidebar.header("Camera Controls")
    
    st.warning("Live camera feed will only work when running this app on your local computer.", icon="⚠️")
    
    camera_index = st.sidebar.selectbox("Select Camera", options=[0, 1, 2], index=1,
                                        help="Select the camera source index. 0 is usually the built-in webcam.")
    
    save_video_checkbox = st.sidebar.checkbox("Save annotated video",
                                            help="Check this to save the output video.",
                                            key="save_video_live")

    start_button = st.sidebar.button("Start Camera", type="primary")
    stop_button = st.sidebar.button("Stop Camera")

    if start_button:
        st.session_state.stop = False
        st.session_state.save_video = save_video_checkbox
        # Clean up any previous video file
        if st.session_state.video_file_path and os.path.exists(st.session_state.video_file_path):
            os.remove(st.session_state.video_file_path)
        st.session_state.video_file_path = None
        st.rerun()

    if stop_button:
        st.session_state.stop = True
        st.rerun()

    # --- Live Camera Detection Loop ---
    if not st.session_state.stop:
        cap = cv2.VideoCapture(camera_index)
        video_writer = None
        tfile_path = None

        if not cap.isOpened():
            st.error(f"Error: Could not open camera with index {camera_index}.")
        else:
            st.info("Camera is running... Press 'Stop Camera' in the sidebar to end.")
            
            # Setup video writer
            if st.session_state.save_video:
                try:
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    if fps == 0: fps = 20
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                        tfile_path = tfile.name
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(tfile_path, fourcc, float(fps), (frame_width, frame_height))
                    st.sidebar.info(f"Saving video to: {tfile_path}")
                except Exception as e:
                    st.sidebar.error(f"Failed to initialize video writer: {e}")
                    st.session_state.save_video = False

            while not st.session_state.stop:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Camera feed ended.")
                    st.session_state.stop = True
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb, conf=confidence, verbose=False)
                annotated_frame_rgb = results[0].plot()
                
                frame_placeholder.image(annotated_frame_rgb, caption="Real-Time Detection", use_column_width=True)
                
                if video_writer:
                    try:
                        annotated_frame_bgr = cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR)
                        video_writer.write(annotated_frame_bgr)
                    except Exception as e:
                        st.sidebar.error(f"Error writing video frame: {e}")
                        st.session_state.stop = True
            
            cap.release()
            if video_writer:
                video_writer.release()
                st.session_state.video_file_path = tfile_path
                st.sidebar.success("Video saving complete.")
            
            if st.session_state.stop:
                frame_placeholder.empty()
                st.info("Camera stopped.")
                st.rerun()

    # --- Download Button for Saved Live Video ---
    if st.session_state.video_file_path and os.path.exists(st.session_state.video_file_path):
        st.sidebar.markdown("---")
        st.sidebar.header("Download")
        try:
            with open(st.session_state.video_file_path, 'rb') as f:
                st.sidebar.download_button(
                    label="Download Annotated Video",
                    data=f,
                    file_name="annotated_live_video.mp4",
                    mime="video/mp4"
                )
        except Exception as e:
            st.sidebar.error(f"Error reading video file for download: {e}")

elif source_option == "Upload a file (Image/Video)":
    
    uploaded_file = st.file_uploader("Upload an image or video file", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])
    
    if uploaded_file:
        file_type = uploaded_file.type.split('/')[0]
        
        # --- Process Image File ---
        if file_type == "image":
            try:
                image = Image.open(uploaded_file)
                image_rgb = image.convert("RGB")
                
                with st.spinner("Processing image..."):
                    results = model(image_rgb, conf=confidence, verbose=False)
                    annotated_image_rgb = results[0].plot()
                
                st.image(annotated_image_rgb, caption="Detection Results", use_column_width=True)
                
                # Offer download for annotated image
                im_bgr = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".png", im_bgr)
                
                st.download_button(
                    label="Download Annotated Image",
                    data=buf.tobytes(),
                    file_name=f"annotated_{uploaded_file.name}",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Error processing image: {e}")

        # --- Process Video File ---
        elif file_type == "video":
            try:
                # Save uploaded video to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(uploaded_file.read())
                    tfile_path_in = tfile.name
                
                cap = cv2.VideoCapture(tfile_path_in)
                if not cap.isOpened():
                    st.error("Error: Could not open uploaded video file.")
                else:
                    # Prepare output video writer
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps == 0: fps = 20
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile_out:
                        tfile_path_out = tfile_out.name
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(tfile_path_out, fourcc, float(fps), (frame_width, frame_height))
                    
                    st.info("Processing video file... This may take a moment.")
                    
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress_bar = st.progress(0, text="Processing...")

                    frame_count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = model(frame_rgb, conf=confidence, verbose=False)
                        annotated_frame_rgb = results[0].plot()
                        
                        annotated_frame_bgr = cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR)
                        video_writer.write(annotated_frame_bgr)
                        
                        frame_count += 1
                        progress_bar.progress(frame_count / total_frames, text=f"Processing frame {frame_count}/{total_frames}")

                    cap.release()
                    video_writer.release()
                    progress_bar.empty()
                    st.success("Video processing complete!")
                    
                    # Display the processed video
                    st.video(tfile_path_out)
                    
                    # Offer download
                    with open(tfile_path_out, 'rb') as f:
                        st.download_button(
                            label="Download Annotated Video",
                            data=f,
                            file_name=f"annotated_{uploaded_file.name}",
                            mime="video/mp4"
                        )
                    
                    # Clean up temp files
                    os.remove(tfile_path_in)
                    os.remove(tfile_path_out)
            
            except Exception as e:
                st.error(f"Error processing video: {e}")
                # Clean up temp files on error
                if 'tfile_path_in' in locals() and os.path.exists(tfile_path_in):
                    os.remove(tfile_path_in)
                if 'tfile_path_out' in locals() and os.path.exists(tfile_path_out):
                    os.remove(tfile_path_out)

else:
    frame_placeholder.info("Select an input source from the sidebar to begin.")

st.markdown("---")
st.sidebar.markdown("---")
st.sidebar.markdown("**About:** This app uses YOLOv8 to perform object detection on uploaded files or a live webcam feed.")
