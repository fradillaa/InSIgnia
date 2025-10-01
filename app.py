# 1. Import streamlit and set page config
import streamlit as st

# Import other necessary libraries
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2, os, glob
from ultralytics import YOLO
import azure.cognitiveservices.speech as speechsdk
from PIL import Image
import tempfile
from streamlit_audiorecorder import audiorecorder
from io import BytesIO
import openai
from pydub import AudioSegment
from dotenv import load_dotenv
import os
import html
import time

# --- Configuration and Initialization ---
load_dotenv() # Load environment variables

# Set FFmpeg path (ensure this path is correct for your deployment)
# This assumes ffmpeg is in a specific relative path. For production, consider adding to PATH or use absolute path.
AudioSegment.converter = os.path.join(os.getcwd(), "ffmpeg-7.1.1-essentials_build", "ffmpeg-7.1.1-essentials_build", "bin", "ffmpeg.exe")
if not os.path.exists(AudioSegment.converter):
    st.error(f"FFmpeg converter not found at: {AudioSegment.converter}. Please check the path and ensure FFmpeg is installed.")
    st.stop() # Stop the app if ffmpeg is not found

# Azure OpenAI Configuration
AZURE_OPEN_AI_API_KEY = os.getenv("AZURE_OPEN_AI_API_KEY")
AZURE_OPEN_AI_ENDPOINT = os.getenv("AZURE_OPEN_AI_ENDPOINT")
AZURE_OPEN_AI_API_VERSION = os.getenv("AZURE_OPEN_AI_API_VERSION")

# Azure Speech Service Configuration
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_SPEECH_VOICE = os.getenv("AZURE_SPEECH_VOICE", 'id-ID-ArdiNeural') # Default voice for Indonesian

# Initialize Azure OpenAI Client
openai_client = openai.AzureOpenAI(
    api_version=AZURE_OPEN_AI_API_VERSION,
    azure_endpoint=AZURE_OPEN_AI_ENDPOINT,
    api_key=AZURE_OPEN_AI_API_KEY
)

# Load YOLO model
try:
    model = YOLO("best.pt")
except Exception as e:
    st.error(f"Error loading YOLO model: {e}. Ensure 'best.pt' is in the root directory.")
    st.stop()

# Initialize Azure Speech Config
speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
speech_config.speech_synthesis_voice_name = AZURE_SPEECH_VOICE
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

# Set Streamlit page configuration
st.set_page_config(
    page_title="InSignia: Jembatan Komunikasi Inklusif",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
# Mapping class index to letters A-Y (without J, Z)
# Mapping class index to letters A-Y (without J, Z)
def get_class_mapping():
    import string
    letters = [c for c in string.ascii_uppercase if c not in ['J', 'Z']]
    return {i: l for i, l in enumerate(letters)}

# Load example images from dataset
@st.cache_data
def load_label_images(dataset_folder="train"):
    label_map = {}
    label_folder = os.path.join(dataset_folder, "labels")
    image_folder = os.path.join(dataset_folder, "images")

    for label_file in glob.glob(os.path.join(label_folder, "*.txt")):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            if not lines: continue
            class_id = lines[0].split()[0]
            if class_id not in label_map:
                img_name = os.path.basename(label_file).replace(".txt", ".jpg")
                img_path = os.path.join(image_folder, img_name)
                if os.path.exists(img_path):
                    label_map[class_id] = img_path
    return label_map

# Webcam Real-Time Detection
class SignLanguageDetector(VideoTransformerBase):
    def _init_(self):
        self.detected_text = ""
        self.last_label = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror
        results = model.predict(img, imgsz=640, conf=st.session_state.get("detection_threshold", 0.6), verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = get_class_mapping().get(cls, "?")

            if label != self.last_label:
                self.detected_text += label
                self.last_label = label

            # Warna ungu (BGR: 255, 0, 255), dan ketebalan garis 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 4)

            # Ukuran font diperbesar (1.5) dan ketebalan teks ditingkatkan (3)
            cv2.putText(img, label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)

        if len(results.boxes) == 0:
            self.last_label = ""

        return img

# --- Modern CSS Styling ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css") # Load our custom stylesheet

# Initialize session state for navigation and dynamic content
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Beranda"
if 'sidebar_expanded' not in st.session_state:
    st.session_state.sidebar_expanded = True
if 'detected_sign_text' not in st.session_state:
    st.session_state.detected_sign_text = ""
if 'detected_text' not in st.session_state:
    st.session_state.detected_text = ""
if 'chatbot_messages' not in st.session_state:
    st.session_state.chatbot_messages = [{"role": "assistant", "content": "Halo! Saya InSignia Bot, siap membantu Anda belajar dan berkomunikasi tentang Bahasa Isyarat SIBI. Ada yang bisa saya bantu?"}]
if 'show_fps_camera' not in st.session_state:
    st.session_state.show_fps_camera = True
if 'detection_threshold' not in st.session_state:
    st.session_state.detection_threshold = 0.6

with st.sidebar:
    # Logo and App Title
    st.markdown("""
    <div style="text-align: center; margin: 1.5rem 0 2.5rem 0;">
        <div style="background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%); width: 80px; height: 80px; border-radius: 16px; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem auto; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <span style="font-size: 2.5rem;">🤟</span>
        </div>
        <h2 style="color: var(--primary-dark); margin-bottom: 0.25rem; font-weight: 700;">InSignia</h2>
        <p style="color: var(--text-light); font-size: 0.9rem; margin-bottom: 0;">AI Deteksi Bahasa Isyarat SIBI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Menu
    menu_items = [
        {"icon": "🏠", "label": "Beranda", "key": "home"},
        {"icon": "🌟", "label": "Fitur Unggulan", "key": "features"},
        {"icon": "📷", "label": "Deteksi SIBI", "key": "detection"},
        {"icon": "📚", "label": "Kamus SIBI", "key": "dictionary"},
        {"icon": "🎤", "label": "Speech to SIBI", "key": "speech"},
        {"icon": "💬", "label": "Chatbot", "key": "chatbot"}
    ]
    
    selected_page = st.session_state.current_page.split()[-1]
    
    for item in menu_items:
        if st.button(f"{item['icon']} {item['label']}", 
                    key=item['key'],
                    use_container_width=True,
                    type="primary" if selected_page == item['label'] else "secondary"):
            st.session_state.current_page = f"{item['icon']} {item['label']}"
            st.rerun()
    
    st.markdown("---")
    
    # User Guide
    with st.expander("📖 Panduan Pengguna", expanded=False):
        st.markdown("""
        <div style="font-size: 0.85rem;">
            <p><strong>1. Deteksi SIBI:</strong> Aktifkan kamera dan tunjukkan isyarat SIBI</p>
            <p><strong>2. Kamus SIBI:</strong> Pelajari huruf dan kata dalam SIBI</p>
            <p><strong>3. Speech to SIBI:</strong> Ucapkan kata untuk melihat visual SIBI</p>
            <p><strong>4. Chatbot:</strong> Tanya apa saja tentang SIBI</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; font-size: 0.8rem; color: var(--text-light); margin-top: 2rem;">
        <p>Versi 1.0.0</p>
        <p>© 2024 Tim InSignia</p>
    </div>
    """, unsafe_allow_html=True)

# --- Page Content Functions ---
def landing_page():
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
        <div class="hero">
            <h1 style="margin-bottom: 1rem;"><span class="gradient-text">InSignia</span></h1>
            <h2 style="font-size: 2.5rem; font-weight: 700; margin-top: 0; color: var(--primary-dark);">
                Real-time SIBI Detection for Inclusive Education
            </h2>
            <p style="font-size: 1.3rem; line-height: 1.8; margin-bottom: 2rem; opacity: 0.9;">
                Platform AI inovatif untuk deteksi Bahasa Isyarat SIBI real-time, menjembatani komunikasi inklusif bagi penyandang disabilitas Rungu Wicara dan masyarakat umum.
            </p>
            <div style="margin-top: 3rem;">
                <span class="badge" style="background-color: var(--primary);">Inklusif</span>
                <span class="badge" style="background-color: #28A745;">Real-time</span>
                <span class="badge" style="background-color: var(--secondary);">Mudah Digunakan</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 3rem;'>", unsafe_allow_html=True)
        if st.button("🚀 Mulai jelajahi InSignia", key="start_button_landing", use_container_width=True, type="primary"):
            st.session_state.current_page = "🌟 Fitur Unggulan"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.image("WhatsApp Image 2025-06-01 at 03.24.53_e4edf93b.jpg", use_container_width=True, caption="Komunikasi Tanpa Batas")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div class="about-section-text">
            <h1>Tentang <span class="gradient-text">InSignia</span></h1>
            <p>
                InSignia hadir sebagai <b>solusi AI terdepan</b> yang memfasilitasi komunikasi inklusif melalui <b>deteksi Bahasa Isyarat SIBI secara real-time</b>. Kami hadir untuk mengatasi hambatan komunikasi yang dialami oleh lebih dari <b>2,5 juta penyandang disabilitas pendengaran di Indonesia</b>, memberdayakan interaksi yang setara dan bermakna.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='vision-tech-grid'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            <div class="card">
                <h3>Visi Kami</h3>
                <p style="line-height: 1.8;">
                    Mewujudkan dunia yang <b>sepenuhnya inklusif</b>, di mana <b>Bahasa Isyarat SIBI mudah dipahami</b> oleh siapa saja, menjembatani kesenjangan komunikasi demi interaksi yang setara dan bermartabat.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h3>Teknologi Kami</h3>
                <p style="line-height: 1.8;">
                    Didukung <b>YOLO Object Detection</b> dan <b>Azure Speech Recognition</b>, InSignia mampu menerjemahkan Bahasa Isyarat SIBI dengan <b>akurasi tinggi dan kecepatan real-time</b>. Kami memanfaatkan kekuatan inti dari <b>Computer Vision, Artificial Intelligence, dan Natural Language Processing</b> untuk solusi ini.
                </p>
                <div style="display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 1rem;">
                    <span class="badge" style="background-color: var(--primary-light);">Computer Vision</span>
                    <span class="badge" style="background-color: var(--accent); color: var(--text-color);">AI</span>
                    <span class="badge" style="background-color: var(--secondary);">NLP</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div style="text-align: center; margin: 4rem 0;">
            <h1>✨ Keunggulan <span class="gradient-text">InSignia</span></h1>
            <p style="color: var(--text-light); font-size: 1.2rem;">Solusi lengkap dan terdepan untuk komunikasi inklusif.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='how-it-works-grid features-grid'>", unsafe_allow_html=True)
        cols = st.columns(3)
        features = [
            {"icon": "⚡", "title": "Deteksi Real-time", "desc": "Mendeteksi dan menerjemahkan Bahasa Isyarat SIBI secara instan dengan model YOLO terbaru, memberikan respons cepat untuk komunikasi yang lancar."},
            {"icon": "🤖", "title": "Kecerdasan Buatan Canggih", "desc": "Ditenagai oleh teknologi AI mutakhir dari Azure, memastikan akurasi dan keandalan yang luar biasa dalam setiap terjemahan."},
            {"icon": "🌐", "title": "Solusi Multi-Modal", "desc": "Mendukung input visual (kamera), suara (mikrofon), dan teks, serta dilengkapi Chatbot interaktif untuk pengalaman komunikasi yang komprehensif."}
        ]
        
        for i, feature in enumerate(features):
            with cols[i]:
                st.markdown(f"""
                <div class="card">
                    <div style="font-size: 3rem; margin-bottom: 1.5rem; color: var(--primary);">{feature['icon']}</div>
                    <h3>{feature['title']}</h3>
                    <p style="line-height: 1.7; color: var(--text-light); opacity: 0.9;">{feature['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div style="text-align: center; margin: 4rem 0;">
            <h1>🛠 Cara Kerja <span class="gradient-text">InSignia</span></h1>
            <p style="color: var(--text-light); opacity: 0.8; font-size: 1.2rem;">Proses yang efisien untuk komunikasi yang kompleks.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='how-it-works-grid'>", unsafe_allow_html=True)
        steps = st.columns(3)
        step_data = [
            {"num": "1", "title": "Input", "desc": "Pengguna memasukkan bahasa isyarat melalui kamera atau suara melalui mikrofon."},
            {"num": "2", "title": "Proses AI", "desc": "Sistem AI canggih kami mengenali isyarat visual atau ucapan, lalu menerjemahkannya secara cerdas."},
            {"num": "3", "title": "Output", "desc": "Hasil terjemahan ditampilkan secara instan dalam format teks yang mudah dipahami atau visual isyarat."}
        ]
        
        for i, step in enumerate(step_data):
            with steps[i]:
                st.markdown(f"""
                <div class="card">
                    <div style="background: var(--primary-light); width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-bottom: 1.5rem; color: var(--white); font-size: 2rem; font-weight: bold; box-shadow: 0 5px 15px rgba(var(--primary-light-rgb), 0.3);">
                        {step['num']}
                    </div>
                    <h3>{step['title']}</h3>
                    <p style="line-height: 1.7; color: var(--text-light); opacity: 0.9;">{step['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div style="text-align: center; margin: 4rem 0;">
            <h1>📢 Testimoni <span class="gradient-text">Pengguna</span></h1>
            <p style="color: var(--text-light); opacity: 0.8; font-size: 1.2rem;">Apa kata mereka yang telah merasakan dampak positif InSignia.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='how-it-works-grid'>", unsafe_allow_html=True)
        testimonials_cols = st.columns(3)
        testimonial_data = [
            {"name": "Budi Santoso", "role": "Guru SLB", "quote": "InSignia sangat membantu siswa saya dalam belajar bahasa isyarat. Prosesnya jadi lebih interaktif dan menyenangkan, mendorong mereka untuk lebih aktif berkomunikasi. Benar-benar alat yang revolusioner di kelas."},
            {"name": "Siti Aminah", "role": "Profesional HRD", "quote": "Sebagai HRD, saya sangat menghargai kemudahan komunikasi dengan rekan tuli di perusahaan. Antarmuka InSignia yang intuitif telah meningkatkan inklusivitas dan kolaborasi tim secara signifikan."},
            {"name": "Dr. Rina Dewi", "role": "Dokter Umum", "quote": "Memberikan pelayanan kesehatan yang inklusif adalah prioritas. InSignia adalah alat vital yang memungkinkan saya berinteraksi lebih efektif dengan pasien tunarungu, memastikan mereka mendapatkan penanganan yang layak dan nyaman."}
        ]
        
        for i, testimonial in enumerate(testimonial_data):
            with testimonials_cols[i]:
                st.markdown(f"""
                <div class="testimonial card">
                    <p style="font-style: italic; margin-bottom: 1.5rem; line-height: 1.7; color: var(--text-color);">
                        "{testimonial['quote']}"
                    </p>
                    <div>
                        <p style="font-weight: bold; color: var(--primary); margin-bottom: 0.25rem; font-size: 1.1rem;">{testimonial['name']}</p>
                        <p style="font-size: 0.95rem; color: var(--text-light); opacity: 0.8;">{testimonial['role']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div style="text-align: center; margin: 4rem 0;">
            <h1 style="margin-bottom: 1.5rem;">Siap Memulai Perjalanan <span class="gradient-text">Inklusivitas Anda?</span></h1>
            <p style="font-size: 1.2rem; color: var(--text-light); opacity: 0.9; margin-bottom: 3rem; max-width: 700px; margin-left: auto; margin-right: auto;">
                Bergabunglah dengan ribuan pengguna yang telah merasakan manfaat InSignia dalam memecahkan hambatan komunikasi dan menciptakan dunia yang lebih setara.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Mulai Sekarang", key="start_button_cta", use_container_width=True, type="primary"):
                st.session_state.current_page = "🌟 Fitur Unggulan"
                st.rerun()
            st.markdown("<p style='text-align: center; margin-top: 1rem; font-size: 0.9rem; color: var(--text-light);'>Tidak perlu instalasi, langsung akses dari browser Anda.</p>", unsafe_allow_html=True)

def features_page():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h1 style="color: var(--primary-dark);">🌟 Jelajahi <span class="gradient-text">Fitur InSignia</span></h1>
        <p style="color: var(--text-light); font-size: 1.1rem;">Temukan bagaimana InSignia memberdayakan komunikasi inklusif.</p>
    </div>
    """, unsafe_allow_html=True)

    feature_data = [
        {"icon": "📷", "title": "Deteksi Real-time", "description": "Deteksi Bahasa Isyarat SIBI secara langsung melalui kamera perangkat Anda dengan akurasi tinggi, mengubah isyarat menjadi teks secara instan.", "key": "open_detect", "page": "📷 Deteksi", "color": "#4A63E0"}, # primary
        {"icon": "📚", "title": "Kamus SIBI Interaktif", "description": "Pelajari Bahasa Isyarat SIBI dengan panduan visual lengkap, ilustrasi interaktif, dan contoh penggunaan untuk memperkaya pemahaman Anda.", "key": "open_dict", "page": "📚 Kamus", "color": "#EE5F8E"}, # secondary
        {"icon": "🎤", "title": "Speech to Visual", "description": "Konversikan ucapan Anda menjadi visual Bahasa Isyarat SIBI, memungkinkan komunikasi dua arah yang lancar antara pengguna bahasa isyarat dan teman bicara mereka.", "key": "open_speech", "page": "🎤 Speech to Visual", "color": "#7209B7"}, # purple
        {"icon": "💬", "title": "Chatbot InSignia", "description": "Dapatkan bantuan instan, informasi mendalam, dan panduan praktis seputar Bahasa Isyarat SIBI dan komunikasi inklusif dari chatbot cerdas kami.", "key": "open_chat", "page": "💬 Chatbot", "color": "#FFD23F"} # accent
    ]

    st.markdown("<div class='how-it-works-grid features-grid'>", unsafe_allow_html=True)
    rows = [feature_data[i:i + 2] for i in range(0, len(feature_data), 2)]
    for row in rows:
        cols = st.columns(len(row))
        for i, feature in enumerate(row):
            with cols[i]:
                st.markdown(f"""
                <div class="card" style="border-top: 5px solid {feature['color']};">
                    <div style="font-size: 3rem; margin-bottom: 1.5rem; color: {feature['color']};">{feature['icon']}</div>
                    <h3>{feature['title']}</h3>
                    <p style="margin-bottom: 1.5rem; color: var(--text-light); line-height: 1.7;">{feature['description']}</p>
                    <div style="margin-top: auto; width: 100%;"> </div>
                </div>
                """, unsafe_allow_html=True)
                # The button is intentionally placed outside the markdown string to ensure Streamlit renders it as a proper widget.
                if st.button(f"Buka {feature['title'].split(' ')[0]}", key=feature['key'], use_container_width=True, type="primary"):
                    st.session_state.current_page = feature['page']
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)
    if st.button("← Kembali ke Beranda", key="back_features_page_bottom", use_container_width=True, type="secondary"):
        st.session_state.current_page = "🏠 Beranda"
        st.rerun()

def detection_page():
    if st.button("← Kembali", key="back_from_detection", type="secondary"):
        st.session_state.current_page = "🌟 Fitur Unggulan"
        st.rerun()
    
    with st.container():
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: var(--primary-color);">📷 Deteksi Bahasa Isyarat</h1>
            <p style="color: var(--text-color);">Aktifkan kamera dan pastikan tangan terlihat jelas di area kamera.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("⚙ Pengaturan Deteksi Kamera", expanded=False):
            st.session_state.show_fps_camera = st.checkbox("Tampilkan FPS di Kamera", value=st.session_state.show_fps_camera)
            st.session_state.detection_threshold = st.slider("Threshold Deteksi (Confidence)", 0.0, 1.0, st.session_state.detection_threshold, 0.05)
            st.info("Atur threshold untuk menyesuaikan sensitivitas deteksi. Nilai lebih tinggi mengurangi deteksi palsu.", icon="ℹ")
        
        col_cam, col_text = st.columns([2, 1])
        with col_cam:
            st.markdown("<h3>Live Kamera Deteksi</h3>", unsafe_allow_html=True)
            ctx = webrtc_streamer(
                key="sign-lang",
                video_transformer_factory=SignLanguageDetector,
                media_stream_constraints={
                    "video": {
                        "width": {"min": 640, "ideal": 1280, "max": 1920},
                        "height": {"min": 480, "ideal": 720, "max": 1080},
                        "frameRate": {"ideal": 30, "max": 60},
                        "facingMode": "user"
                    },
                    "audio": False
                },
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # Add STUN server for NAT traversal
                async_transform=True # Enable async processing for smoother video
            )
            
            if ctx.state.playing:
                st.success("✅ Kamera aktif, deteksi sedang berjalan!", icon="🎉")
            else:
                st.warning("⚠ Kamera belum diaktifkan. Klik 'START' untuk memulai deteksi.", icon="⚠")

        with col_text:
            st.markdown("""
            <div style="
                background-color: #f0f7ff;
                padding: 1em;
                border-radius: 0.5em;
                border-left: 4px solid #1e88e5;
                margin-bottom: 1em;
            ">
                <h3><strong>💡Petunjuk Penggunaan</strong></h3>
                <ol style="padding-left: 1.5rem; line-height: 1.8;">
                    <li><b>Pencahayaan Cukup:</b> Pastikan area tangan Anda terang dan tidak ada bayangan.</li>
                    <li><b>Posisikan di Tengah:</b> Letakkan tangan Anda di tengah bingkai kamera.</li>
                    <li><b>Latar Belakang Kontras:</b> Gunakan latar belakang polos atau kontras agar tangan lebih mudah dikenali.</li>
                    <li><b>Gerakan Jelas:</b> Lakukan gerakan isyarat dengan jelas dan terpisah.</li>
                    <li><b>Reset Teks:</b> Teks deteksi akan terakumulasi. Gunakan tombol "Hapus Teks Deteksi" untuk memulai ulang.</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.detected_sign_text:
                if st.button("Terjemahkan ke Suara (Preview)", key="translate_to_speech", use_container_width=True):
                    text_to_speak = st.session_state.detected_sign_text
                    if text_to_speak:
                        with st.spinner("Mengonversi teks ke suara..."):
                            try:
                                result = speech_config.speech_synthesizer.speak_text_async(text_to_speak).get()
                                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                                    st.success("Audio terjemahan siap.")
                                    # For playing audio in Streamlit, usually you'd save it and play it.
                                    # For a simple demo, Azure's SDK plays it directly if speaker is enabled.
                                else:
                                    st.error(f"Gagal mengonversi teks ke suara: {result.reason}")
                            except Exception as e:
                                st.error(f"Error dalam konversi teks ke suara: {e}")
                    else:
                        st.warning("Tidak ada teks untuk diterjemahkan ke suara.")

def dictionary_page():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: var(--primary-dark);">📚 <span class="gradient-text">Kamus Bahasa Isyarat SIBI</span></h1>
        <p style="color: var(--text-light); font-size: 1.1rem;">Telusuri dan pelajari Bahasa Isyarat SIBI dengan panduan visual interaktif.</p>
    </div>
    """, unsafe_allow_html=True)
    
    search_term = st.text_input("🔍 Cari huruf atau kata", placeholder="Contoh: A, B, Halo, Terima Kasih", key="dict_search_input").strip()
    
    class_map = get_class_mapping()
    image_map = load_label_images()

    filtered_items = []
    if search_term:
        # Prioritize exact letter match for single character search
        if len(search_term) == 1 and search_term.isalpha():
            for class_id, letter in class_map.items():
                if search_term.upper() == letter:
                    filtered_items.append((class_id, letter))
        else:
            # For multi-character search (implies searching for a word, which needs a different dataset)
            # For now, we only support alphabet lookup.
            st.info("Fitur pencarian kata (selain alfabet tunggal) belum tersedia. Silakan cari per huruf (A-Y).", icon="ℹ")
            # Fallback to general alphabet search if no exact single letter match for multi-char input
            for class_id, letter in class_map.items():
                if search_term.upper() in letter: # This will still only match individual letters
                    filtered_items.append((class_id, letter))
    else:
        filtered_items = list(class_map.items())

    st.markdown("### Alfabet Bahasa Isyarat SIBI")
    st.markdown("<p style='color: var(--text-light);'>Berikut adalah daftar lengkap huruf dalam Sistem Isyarat Bahasa Indonesia (SIBI) disertai visual:</p>", unsafe_allow_html=True)
    
    if not filtered_items:
        st.warning("Tidak ada hasil ditemukan untuk pencarian Anda.", icon="⚠")
    else:
        cols_per_row = 6
        items = sorted(filtered_items, key=lambda x: x[1])
        num_items = len(items)
        num_rows = (num_items + cols_per_row - 1) // cols_per_row

        st.markdown("<div class='how-it-works-grid'>", unsafe_allow_html=True)
        for r in range(num_rows):
            cols = st.columns(cols_per_row)
            for i in range(cols_per_row):
                idx = r * cols_per_row + i
                if idx < num_items:
                    class_id, letter = items[idx]
                    img_path = image_map.get(str(class_id))
                    with cols[i]:
                        st.markdown(f"""
                        <div class="dictionary-card">
                            <h3 style="margin-top: 0; color: var(--primary-dark);">{letter}</h3>
                            """, unsafe_allow_html=True)
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        else:
                            st.markdown("<p style='color: var(--text-light); font-size: 0.9rem;'>(Gambar tidak tersedia)</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)
    if st.button("← Kembali", key="back_from_dictionary_bottom", use_container_width=True, type="secondary"):
        st.session_state.current_page = "🌟 Fitur Unggulan"
        st.rerun()

def speech_page():
    if st.button("← Kembali", key="back_from_speech", type="secondary"):
        st.session_state.current_page = "🌟 Fitur Unggulan"
        st.rerun()
    
    with st.container():
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: var(--primary-color);">🎤 Speech to Visual</h1>
            <p style="color: var(--text-color);">Konversi ucapan Anda menjadi visual bahasa isyarat SIBI.</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🎙 Rekam Suara", "📂 Upload Audio"])
        
        detected_text = None
        audio_path = None
        
        with tab1:
            st.markdown("### 🔴 Rekam Suara Anda")
            audio = audiorecorder("🎙 Mulai Rekam", "⏹ Berhenti Rekam", key="recorder")
            
            detected_text = None
            audio_path = None
            
            if audio is not None and len(audio) > 0:
                st.audio(audio.export().read(), format="audio/wav")
                if st.button("🔊 Proses Rekaman", key="process_recording", use_container_width=True):
                    with st.spinner("🔄 Memproses rekaman..."):                        
                        if isinstance(audio, AudioSegment):
                            audio_segment = audio
                        else:
                            audio_segment = AudioSegment.from_file(BytesIO(audio), format="webm")

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                            audio_segment.export(f.name, format="wav")
                            audio_path = f.name
                        detected_text = None  # reset detected_text untuk input baru
                        
                        if audio_path:
                            audio_input = speechsdk.AudioConfig(filename=audio_path)
                            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
                            result = recognizer.recognize_once()
                        
                            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                                detected_text = result.text.upper()
                                st.session_state.detected_text = detected_text
                                st.success(f"🗣 Teks Terdeteksi: {detected_text}")
                            else:
                                st.error(f"Gagal mengenali suara. Reason: {result.reason}")
            
        with tab2:
            st.markdown("### 📂 Upload File Audio")
            uploaded_file = st.file_uploader("Pilih file audio (.wav/.mp3)", type=["wav", "mp3"], label_visibility="collapsed")
            
            if uploaded_file is not None:
                st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
                
                if st.button("🔊 Proses File Audio", key="process_upload", use_container_width=True):
                    with st.spinner("🔄 Memproses file audio..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                            f.write(uploaded_file.read())
                            audio_path = f.name
                        
                        audio_input = speechsdk.AudioConfig(filename=audio_path)
                        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
                        result = recognizer.recognize_once()
                        
                        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                            detected_text = result.text.upper()
                            st.session_state.detected_text = detected_text
                            st.success(f"🗣 Teks Terdeteksi: {detected_text}")
                        else:
                            st.error(f"Gagal mengenali suara. Reason: {result.reason}")

        # Display sign language visuals based on detected/entered text
        st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)
        st.markdown("### 🖼 Visual Bahasa Isyarat dari Teks")

        # Create two columns for input and button
        col1, col2 = st.columns([4, 1])
        with col1:
            input_text_for_visuals = st.text_input(
                "Atau masukkan teks manual untuk melihat visual SIBI:",
                value=st.session_state.detected_text if 'detected_text' in st.session_state else "",
                key="text_input_for_visuals",
                placeholder="Contoh: HALO, TERIMA KASIH"
            )
        with col2:
            st.write("")  # For vertical alignment
            process_text = st.button("🔍 Tampilkan Visual", use_container_width=True)

        # Display processing message
        if process_text and input_text_for_visuals:
            with st.spinner("🔄 Memproses teks dan menyiapkan visual SIBI..."):
                time.sleep(0.5)  # Simulate processing time
                
                # Process the text and display visuals
                class_map = get_class_mapping()
                inv_class_map = {v: str(k) for k, v in class_map.items()}
                image_map = load_label_images()
                
                # Clean and prepare the text
                processed_text = input_text_for_visuals.strip().upper()
                st.session_state.detected_text = processed_text  # Store for persistence
                
                st.markdown("### 👐 Visualisasi Bahasa Isyarat")
                
                # Display the original text
                st.markdown(f"""
                <div style="background: var(--card-bg); padding: 1rem; border-radius: var(--border-radius); 
                            margin-bottom: 1rem; box-shadow: var(--box-shadow);">
                    <p style="margin: 0; font-weight: 500;">Teks yang diproses:</p>
                    <p style="margin: 0; font-size: 1.2rem;">{processed_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Filter valid SIBI characters
                valid_chars = [c for c in processed_text if c in inv_class_map]
                invalid_chars = [c for c in processed_text if c not in inv_class_map and c != ' ']
                
                # Show stats about the text
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.metric("Total Karakter", len(processed_text))
                with stats_col2:
                    st.metric("Huruf SIBI Valid", len(valid_chars))
                
                if invalid_chars:
                    st.warning(f"Karakter berikut tidak memiliki visual SIBI: {', '.join(set(invalid_chars))}")
                
                # Display SIBI signs in a responsive grid
                if valid_chars:
                    st.markdown("#### Huruf SIBI yang Dikenali")
                    
                    # Display 6 signs per row
                    cols_per_row = 6
                    num_rows = (len(valid_chars) + cols_per_row - 1) // cols_per_row
                    
                    for row in range(num_rows):
                        cols = st.columns(cols_per_row)
                        start_idx = row * cols_per_row
                        end_idx = start_idx + cols_per_row
                        row_chars = valid_chars[start_idx:end_idx]
                        
                        for i, char in enumerate(row_chars):
                            with cols[i]:
                                class_id = inv_class_map[char]
                                img_path = image_map.get(class_id)
                                
                                st.markdown(f"""
                                <div style="text-align: center; padding: 0.5rem; margin-bottom: 1rem; 
                                            background: var(--card-bg); border-radius: var(--border-radius); 
                                            box-shadow: var(--box-shadow); transition: var(--transition);">
                                    <h4 style="margin: 0.5rem 0; color: var(--primary-dark);">{char}</h4>
                                """, unsafe_allow_html=True)
                                
                                if img_path and os.path.exists(img_path):
                                    st.image(img_path, use_container_width=True)
                                else:
                                    st.markdown(f"""
                                    <div style="height: 100px; display: flex; align-items: center; 
                                                justify-content: center; background: #F3F4F6; 
                                                border-radius: 8px; margin-bottom: 0.5rem;">
                                        <p style="color: var(--text-light);">Gambar tidak tersedia</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("Tidak ada huruf SIBI yang valid dalam teks yang dimasukkan.")
                    
                # Add space before the next section
                st.markdown("<br><br>", unsafe_allow_html=True)
        elif process_text and not input_text_for_visuals:
            st.warning("Silakan masukkan teks terlebih dahulu")
    

def chatbot_page():
    if st.button("← Kembali", key="back_from_chatbot", type="secondary"):
        st.session_state.current_page = "🌟 Fitur Unggulan"
        st.rerun()

    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: var(--primary-color);">💬 Chatbot <span class="gradient-text">InSignia</span></h1>
        <p style="color: var(--text-color);">Ajukan pertanyaan seputar Bahasa Isyarat SIBI, inklusivitas, atau fitur InSignia.</p>
    </div>
    """, unsafe_allow_html=True)

    # Display chat messages from history
    for message in st.session_state.chatbot_messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-row user-row">
                <div class="chat-message-container user-message">
                    <p>{html.escape(message["content"])}</p>
                </div>
                <div class="chat-avatar" style="background-color: #555;">😎</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-row bot-row">
                <div class="chat-avatar">🤖</div>
                <div class="chat-message-container bot-message">
                    <p>{html.escape(message["content"])}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # User input
    user_query = st.chat_input("Tanyakan sesuatu tentang SIBI atau InSignia...", key="chatbot_input")

    if user_query:        
        st.session_state.chatbot_messages.append({"role": "user", "content": user_query})        
        with st.spinner("🤖 InSignia Bot sedang berpikir..."):
            try:
                # Prepare messages for OpenAI API
                messages_for_api = [
                    {"role": "system", "content": "Anda adalah InSignia Bot, chatbot yang ramah dan informatif. Anda ahli dalam Bahasa Isyarat SIBI, inklusivitas untuk penyandang disabilitas pendengaran, dan fitur-fitur aplikasi InSignia. Berikan jawaban yang membantu, akurat, dan mendorong inklusivitas. Gunakan bahasa Indonesia yang baik dan benar."}
                ] + st.session_state.chatbot_messages[-5:] # Send last 5 messages for context

                response = openai_client.chat.completions.create(
                    model="gpt-4", # Replace with your actual deployed model name (e.g., gpt-35-turbo, gpt-4)
                    messages=messages_for_api,
                    temperature=0.7,
                    max_tokens=500
                )
                bot_response = response.choices[0].message.content
                st.session_state.chatbot_messages.append({"role": "assistant", "content": bot_response})
            except Exception as e:
                st.error(f"Maaf, terjadi kesalahan saat berkomunikasi dengan chatbot: {e}")
                st.session_state.chatbot_messages.append({"role": "assistant", "content": "Maaf, saya tidak bisa memproses permintaan Anda saat ini. Silakan coba lagi nanti."})
        st.rerun()
    
def settings_page():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: var(--primary-color);">⚙ Pengaturan Aplikasi</h1>
        <p style="color: var(--text-color);">Sesuaikan preferensi InSignia Anda.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Pengaturan Deteksi Kamera")
    st.session_state.show_fps_camera = st.checkbox("Tampilkan FPS di halaman Deteksi", value=st.session_state.show_fps_camera, key="settings_show_fps")
    st.session_state.detection_threshold = st.slider("Threshold Deteksi (Confidence Model)", 0.0, 1.0, st.session_state.detection_threshold, 0.05, key="settings_detection_threshold")
    st.info("Threshold deteksi mempengaruhi seberapa yakin model harus sebelum menandai isyarat.", icon="ℹ")

    st.markdown("### Pengaturan Umum")
    # Example of a future setting:
    # st.selectbox("Bahasa Aplikasi", ["Bahasa Indonesia", "English"], key="app_language")
    
    # Reset button for all settings if needed
    if st.button("Reset Pengaturan ke Default", key="reset_settings", type="secondary"):
        st.session_state.show_fps_camera = True
        st.session_state.detection_threshold = 0.6
        st.success("Pengaturan telah direset ke nilai default.")
        st.rerun()


# --- Main Application Logic ---
if st.session_state.current_page == "🏠 Beranda":
    landing_page()
elif st.session_state.current_page == "🌟 Fitur Unggulan":
    features_page()
elif st.session_state.current_page == "📷 Deteksi":
    detection_page()
elif st.session_state.current_page == "📚 Kamus":
    dictionary_page()
elif st.session_state.current_page == "🎤 Speech to Visual":
    speech_page()
elif st.session_state.current_page == "💬 Chatbot":
    chatbot_page()
elif st.session_state.current_page == "⚙ Pengaturan":
    settings_page()