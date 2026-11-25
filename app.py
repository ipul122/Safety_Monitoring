"""Aplikasi Streamlit: deteksi APD (helm & rompi).

Dua mode (dipilih lewat dropdown "Mode" di sidebar kiri):
    1. Images : unggah foto atau pakai gambar contoh, lalu lihat hasil deteksinya.
    2. Video  : deteksi langsung pada video YouTube.

Cara menjalankan:
    streamlit run app.py
"""

import time
from collections import Counter
from pathlib import Path

import cv2
import streamlit as st
import yt_dlp
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Pengaturan
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "src" / "safety_monitoring" / "model" / "best.pt"
IMAGES_DIR = Path(__file__).parent / "images"  # gambar contoh
ALLOWED_TYPES = ["jpg", "jpeg", "png"]

# Pilihan di dropdown sidebar & sumber gambar
MODE_IMAGES = "Images"
MODE_VIDEO = "Video"
SOURCE_UPLOAD = "Unggah gambar"
SOURCE_SAMPLE = "Gambar contoh"

IMAGE_SIZE = 800  # sama dengan ukuran gambar saat training (mode gambar)
VIDEO_IMAGE_SIZE = 640  # lebih ringan; cukup untuk video resolusi 480p

# Video bawaan di dropdown: {nama tampilan: link YouTube}
VIDEO_OPTIONS = {
    "Video 1 - Keselamatan APD": "https://www.youtube.com/watch?v=LL1e55Dnd_g",
    "Video 2 - CCTV area konstruksi": "https://www.youtube.com/watch?v=T40KkFDvHng",
    "Video 3 - Shorts rompi proyek": "https://www.youtube.com/shorts/uKeRYHfQvyg",
}
CUSTOM_VIDEO = "Link YouTube lain..."

# Ambil video dengan sisi terpendek maksimal 480 piksel (video lebar maupun
# vertikal/Shorts): cukup untuk deteksi dan jauh lebih ringan.
YOUTUBE_FORMAT = "bestvideo[ext=mp4]/bestvideo"
YOUTUBE_FORMAT_SORT = ["res:480", "ext:mp4"]
FALLBACK_VIDEO_FPS = 25  # dipakai jika FPS video tidak terbaca

# Ringkasan jumlah: urutan tetap, {nama kelas di model: label yang ditampilkan}
SUMMARY_LABELS = {"helmet": "Helm", "vest": "Vest", "person": "Person"}


# ---------------------------------------------------------------------------
# Fungsi bantu: model & deteksi
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model() -> YOLO:
    """Memuat model YOLO sekali saja agar aplikasi tetap cepat."""
    return YOLO(str(MODEL_PATH))


def detect_objects(model: YOLO, image, confidence: float, image_size: int):
    """Menjalankan deteksi pada satu gambar (PIL) atau satu frame video (array)."""
    return model.predict(image, conf=confidence, imgsz=image_size, verbose=False)[0]


def draw_detections(result) -> Image.Image:
    """Menggambar kotak deteksi pada gambar, lalu mengubahnya ke format RGB."""
    annotated_bgr = result.plot()  # hasil Ultralytics berformat BGR
    return Image.fromarray(annotated_bgr[..., ::-1])  # BGR -> RGB


def count_by_class(result) -> Counter:
    """Menghitung jumlah objek terdeteksi untuk tiap kelas."""
    class_names = result.names
    return Counter(class_names[int(c)] for c in result.boxes.cls)


# ---------------------------------------------------------------------------
# Fungsi bantu: video YouTube
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def get_stream_url(youtube_url: str) -> str:
    """Mengambil alamat stream langsung dari link YouTube (tanpa mengunduh)."""
    options = {
        "format": YOUTUBE_FORMAT,
        "format_sort": YOUTUBE_FORMAT_SORT,
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(youtube_url, download=False)
    return info["url"]


def sample_frames(capture: cv2.VideoCapture, step: int, max_frames: int):
    """Mengambil 1 frame dari setiap `step` frame, sampai `max_frames` frame.

    Frame yang dilewati hanya di-`grab` (tidak didecode penuh) sehingga ringan.
    """
    for _ in range(max_frames):
        for _ in range(step - 1):
            capture.grab()
        success, frame = capture.read()
        if not success:  # video habis
            return
        yield frame


# ---------------------------------------------------------------------------
# Tampilan umum
# ---------------------------------------------------------------------------
def show_sidebar() -> tuple[str, float]:
    """Menampilkan pengaturan di sidebar. Mengembalikan (mode, confidence)."""
    st.sidebar.header("Pengaturan")
    mode = st.sidebar.selectbox("Mode", [MODE_IMAGES, MODE_VIDEO])
    confidence = st.sidebar.slider(
        "Confidence level",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Semakin tinggi nilainya, semakin ketat deteksi "
        "(lebih sedikit objek, tapi lebih yakin).",
    )
    return mode, confidence


def show_summary(result) -> None:
    """Menampilkan jumlah Helm, Vest, dan Person (selalu tampil, 0 jika tidak ada)."""
    counts = count_by_class(result)

    st.subheader("Jumlah terdeteksi")
    for column, (class_name, label) in zip(st.columns(3), SUMMARY_LABELS.items()):
        column.metric(label=label, value=counts[class_name])


# ---------------------------------------------------------------------------
# Mode 1: gambar
# ---------------------------------------------------------------------------
def pick_image() -> Image.Image | None:
    """Meminta pengguna memilih gambar: unggah sendiri atau gambar contoh."""
    source = st.radio(
        "Sumber gambar", [SOURCE_UPLOAD, SOURCE_SAMPLE], horizontal=True
    )

    if source == SOURCE_UPLOAD:
        uploaded_file = st.file_uploader("Unggah gambar", type=ALLOWED_TYPES)
        if uploaded_file is None:
            st.info("Silakan unggah gambar (JPG atau PNG) untuk memulai.")
            return None
        return Image.open(uploaded_file).convert("RGB")

    sample_files = sorted(
        path for path in IMAGES_DIR.glob("*") if path.suffix[1:].lower() in ALLOWED_TYPES
    )
    if not sample_files:
        st.warning(f"Tidak ada gambar contoh di folder '{IMAGES_DIR.name}'.")
        return None
    chosen = st.selectbox("Pilih gambar contoh", sample_files, format_func=lambda p: p.name)
    return Image.open(chosen).convert("RGB")


def run_image_mode(confidence: float) -> None:
    image = pick_image()
    if image is None:
        return

    with st.spinner("Mendeteksi..."):
        result = detect_objects(load_model(), image, confidence, IMAGE_SIZE)

    original_column, result_column = st.columns(2)
    original_column.subheader("Gambar asli")
    original_column.image(image, use_container_width=True)
    result_column.subheader("Hasil deteksi")
    result_column.image(draw_detections(result), use_container_width=True)

    show_summary(result)


# ---------------------------------------------------------------------------
# Mode 2: video YouTube
# ---------------------------------------------------------------------------
def run_video_mode(confidence: float) -> None:
    target_fps = st.sidebar.slider(
        "Frame diproses per detik",
        min_value=1,
        max_value=20,
        value=10,
        help="Makin besar makin halus tapi makin berat. "
        "Turunkan jika terasa lambat (3-5 sudah cukup untuk memantau APD).",
    )
    max_seconds = st.sidebar.slider(
        "Durasi video diproses (detik)",
        min_value=10,
        max_value=300,
        value=60,
        step=10,
    )

    youtube_url = pick_video_url()
    if not st.button("Mulai deteksi", type="primary"):
        st.info("Atur pengaturan di sidebar, lalu klik **Mulai deteksi**.")
        return

    try:
        with st.spinner("Mengambil video dari YouTube..."):
            stream_url = get_stream_url(youtube_url)
    except yt_dlp.utils.DownloadError:
        st.error("Video tidak bisa diambil. Periksa link atau coba lagi nanti.")
        return

    capture = cv2.VideoCapture(stream_url)
    if not capture.isOpened():
        st.error("Video tidak bisa dibuka.")
        return

    try:
        process_video(capture, confidence, target_fps, max_seconds)
    finally:
        capture.release()


def pick_video_url() -> str:
    """Meminta pengguna memilih video bawaan atau mengetik link YouTube sendiri."""
    choice = st.selectbox("Pilih video", [*VIDEO_OPTIONS, CUSTOM_VIDEO])
    if choice == CUSTOM_VIDEO:
        return st.text_input("Link YouTube", placeholder="https://www.youtube.com/watch?v=...")
    return VIDEO_OPTIONS[choice]


def process_video(
    capture: cv2.VideoCapture, confidence: float, target_fps: int, max_seconds: int
) -> None:
    """Mendeteksi frame demi frame dan menampilkannya secara langsung."""
    model = load_model()
    video_fps = capture.get(cv2.CAP_PROP_FPS) or FALLBACK_VIDEO_FPS
    step = max(1, round(video_fps / target_fps))  # ambil 1 dari setiap `step` frame
    seconds_per_frame = 1 / target_fps

    # Jumlah frame yang akan diproses (video pendek bisa lebih kecil dari batas durasi).
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(max_seconds * target_fps)
    total_frames = min(max_frames, video_frames // step) if video_frames > 0 else max_frames

    progress = st.progress(0.0)
    frame_box = st.empty()
    summary_box = st.empty()

    for number, frame in enumerate(sample_frames(capture, step, max_frames), start=1):
        started_at = time.perf_counter()

        result = detect_objects(model, frame, confidence, VIDEO_IMAGE_SIZE)
        frame_box.image(draw_detections(result), use_container_width=True)
        with summary_box.container():
            show_summary(result)
        progress.progress(min(number / total_frames, 1.0))

        # Jaga kecepatan tayang mendekati waktu asli video (tidak terlalu cepat).
        elapsed = time.perf_counter() - started_at
        time.sleep(max(0.0, seconds_per_frame - elapsed))

    st.success("Selesai.")


# ---------------------------------------------------------------------------
# Titik masuk
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Safety Monitoring", page_icon="🦺", layout="wide")
    st.title("🦺 Safety Monitoring")
    st.write("Deteksi helm dan rompi keselamatan pada gambar atau video.")

    mode, confidence = show_sidebar()

    if mode == MODE_VIDEO:
        run_video_mode(confidence)
    else:
        run_image_mode(confidence)


main()
