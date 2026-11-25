"""Aplikasi Streamlit: deteksi APD (helm & rompi).

Dua mode (dipilih lewat dropdown "Mode" di sidebar kiri):
    1. Images : unggah foto atau pakai gambar contoh, lalu lihat hasil deteksinya.
    2. Video  : deteksi langsung pada video contoh (folder videos), link YouTube,
                atau file video unggahan.

Cara menjalankan:
    streamlit run app.py
"""

import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
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
VIDEO_IMAGE_SIZE = 800  # ukuran masuk model untuk video; turunkan ke 640 jika terasa lambat

# Video contoh: semua file di folder ini muncul otomatis di dropdown.
VIDEOS_DIR = Path(__file__).parent / "videos"
CUSTOM_VIDEO = "Link YouTube lain..."
UPLOAD_VIDEO = "Unggah file video"
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]

# Untuk link YouTube: pilih codec H.264 (avc1) karena paling mudah dibaca OpenCV,
# dengan sisi terpendek video maksimal sekian piksel (juga untuk video vertikal).
YOUTUBE_FORMAT = "bestvideo[vcodec^=avc1][ext=mp4]/bestvideo[ext=mp4]/bestvideo"
YOUTUBE_RESOLUTION = 480
FALLBACK_VIDEO_FPS = 25  # dipakai jika FPS video tidak terbaca

# Warna kotak deteksi per kelas (RGB)
BOX_COLORS = {
    "helmet": (255, 0, 0),  # merah
    "person": (255, 255, 0),  # kuning
    "vest": (0, 0, 255),  # biru
    "no-helmet": (255, 140, 0),  # oranye
    "no-vest": (200, 0, 200),  # ungu
}
DEFAULT_BOX_COLOR = (0, 200, 0)  # hijau, untuk kelas lain

# Ringkasan jumlah: urutan tetap, {nama kelas di model: label yang ditampilkan}
SUMMARY_LABELS = {"helmet": "Helm", "vest": "Vest", "person": "Person"}

# Pasangan perlengkapan: (kelas patuh, kelas pelanggaran, nama untuk keterangan)
EQUIPMENT_CHECKS = (
    ("helmet", "no-helmet", "helm"),
    ("vest", "no-vest", "vest"),
)
# Helm/vest dianggap milik seorang person jika minimal sekian bagian
# kotaknya berada di dalam kotak person tersebut.
MIN_INSIDE_RATIO = 0.6


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Pilihan pengguna di sidebar."""

    mode: str
    confidence: float
    match_persons: bool


@dataclass(frozen=True)
class VideoSource:
    """Sumber video yang dipilih pengguna (hanya satu yang terisi)."""

    file: Path | None = None  # video contoh dari folder `videos`
    youtube_url: str = ""  # link YouTube yang diketik pengguna
    upload: object | None = None  # file video yang diunggah pengguna

    @property
    def is_empty(self) -> bool:
        return self.file is None and not self.youtube_url and self.upload is None


@dataclass(frozen=True)
class Detection:
    """Satu objek hasil deteksi."""

    name: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    note: str = ""  # keterangan tambahan (status APD untuk person)


# ---------------------------------------------------------------------------
# Deteksi
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model() -> YOLO:
    """Memuat model YOLO sekali saja agar aplikasi tetap cepat."""
    return YOLO(str(MODEL_PATH))


def detect_objects(model: YOLO, image, settings: Settings, image_size: int) -> list[Detection]:
    """Mendeteksi objek pada satu gambar (PIL) atau satu frame video (array BGR)."""
    result = model.predict(image, conf=settings.confidence, imgsz=image_size, verbose=False)[0]
    detections = [
        Detection(result.names[int(class_id)], float(score), tuple(box))
        for box, class_id, score in zip(
            result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
        )
    ]
    return match_to_persons(detections) if settings.match_persons else detections


# ---------------------------------------------------------------------------
# Mencocokkan helm & vest dengan person
# ---------------------------------------------------------------------------
def inside_ratio(inner: tuple, outer: tuple) -> float:
    """Berapa bagian kotak `inner` (0-1) yang berada di dalam kotak `outer`."""
    left, top = max(inner[0], outer[0]), max(inner[1], outer[1])
    right, bottom = min(inner[2], outer[2]), min(inner[3], outer[3])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return overlap / area if area > 0 else 0.0


def find_owner(item: Detection, persons: list[Detection]) -> int | None:
    """Mencari person pemilik `item`. None jika tidak berada di dalam person mana pun."""
    ratios = [inside_ratio(item.box, person.box) for person in persons]
    if not ratios:
        return None
    best = max(range(len(ratios)), key=ratios.__getitem__)
    return best if ratios[best] >= MIN_INSIDE_RATIO else None


def resolve_conflicts(items: list[Detection]) -> list[Detection]:
    """Jika satu person terdeteksi 'helm' dan 'tanpa helm' sekaligus, ambil skor tertinggi.

    Berlaku juga untuk vest dan 'tanpa vest'.
    """
    for good, bad, _ in EQUIPMENT_CHECKS:
        best_good = max((i.score for i in items if i.name == good), default=0.0)
        best_bad = max((i.score for i in items if i.name == bad), default=0.0)
        if best_good and best_bad:
            loser = bad if best_good >= best_bad else good
            items = [i for i in items if i.name != loser]
    return items


def describe_equipment(items: list[Detection]) -> str:
    """Membuat keterangan status APD seorang person, contoh: 'helm OK | tanpa vest'."""
    names = {item.name for item in items}
    parts = []
    for good, bad, word in EQUIPMENT_CHECKS:
        if good in names:
            parts.append(f"{word} OK")
        elif bad in names:
            parts.append(f"tanpa {word}")
        else:
            parts.append(f"{word} ?")  # belum terdeteksi
    return " | ".join(parts)


def match_to_persons(detections: list[Detection]) -> list[Detection]:
    """Menyaring deteksi: helm/vest hanya dipakai jika berada di dalam seorang person.

    Setiap person juga mendapat keterangan status APD-nya.
    """
    persons = [d for d in detections if d.name == "person"]
    owned_items: list[list[Detection]] = [[] for _ in persons]
    for item in (d for d in detections if d.name != "person"):
        owner = find_owner(item, persons)
        if owner is not None:
            owned_items[owner].append(item)

    matched = []
    for person, items in zip(persons, owned_items):
        items = resolve_conflicts(items)
        matched.append(replace(person, note=describe_equipment(items)))
        matched.extend(items)
    return matched


def count_by_class(detections: list[Detection]) -> Counter:
    """Menghitung jumlah objek terdeteksi untuk tiap kelas."""
    return Counter(d.name for d in detections)


# ---------------------------------------------------------------------------
# Menggambar hasil
# ---------------------------------------------------------------------------
def draw_label(image, text: str, x: int, y: int, color: tuple, font_scale: float) -> None:
    """Menulis label kecil (dengan latar sewarna kotak) di atas kotak deteksi."""
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    top = max(0, y - text_height - baseline)  # jangan keluar dari gambar
    cv2.rectangle(image, (x, top), (x + text_width, top + text_height + baseline), color, -1)

    is_bright = sum(color) / 3 > 150
    text_color = (0, 0, 0) if is_bright else (255, 255, 255)
    cv2.putText(
        image, text, (x, top + text_height), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, text_color, 1, cv2.LINE_AA,
    )


def draw_detections(image_rgb: np.ndarray, detections: list[Detection]) -> Image.Image:
    """Menggambar kotak tipis berwarna per kelas pada gambar."""
    image = image_rgb.copy()
    longest_side = max(image.shape[:2])
    thickness = max(1, round(longest_side / 800))  # tipis; ikut menebal di gambar besar
    font_scale = max(0.4, longest_side / 2000)

    for detection in detections:
        color = BOX_COLORS.get(detection.name, DEFAULT_BOX_COLOR)
        x1, y1, x2, y2 = (int(value) for value in detection.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        text = f"{detection.name} {detection.score:.2f}"
        if detection.note:
            text += f" | {detection.note}"

        # Label person di tepi bawah kotak supaya tidak tertimpa label helm di atas kepala.
        label_y = min(y2, image.shape[0]) if detection.name == "person" else y1
        draw_label(image, text, x1, label_y, color, font_scale)

    return Image.fromarray(image)


# ---------------------------------------------------------------------------
# Fungsi bantu: video YouTube
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def get_stream_url(youtube_url: str, resolution: int) -> str:
    """Mengambil alamat stream langsung dari link YouTube (tanpa mengunduh)."""
    with yt_dlp.YoutubeDL(youtube_options(resolution)) as downloader:
        info = downloader.extract_info(youtube_url, download=False)
    return info["url"]


def download_video(youtube_url: str, resolution: int, folder: str) -> str:
    """Mengunduh video YouTube ke `folder`. Mengembalikan path file hasil unduhan."""
    options = {**youtube_options(resolution), "outtmpl": f"{folder}/video.%(ext)s"}
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(youtube_url, download=True)
        return downloader.prepare_filename(info)


def youtube_options(resolution: int) -> dict:
    """Pengaturan yt-dlp. `resolution` = batas sisi terpendek (juga untuk video vertikal)."""
    return {
        "format": YOUTUBE_FORMAT,
        "format_sort": [f"res:{resolution}"],
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
    }


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
def show_sidebar() -> Settings:
    """Menampilkan pengaturan di sidebar dan mengembalikan pilihan pengguna."""
    st.sidebar.header("Pengaturan")
    mode = st.sidebar.selectbox("Mode", [MODE_IMAGES, MODE_VIDEO])
    confidence = st.sidebar.slider(
        "Confidence level",
        min_value=0.05,
        max_value=1.0,
        value=0.15,
        step=0.05,
        help="Semakin tinggi nilainya, semakin ketat deteksi "
        "(lebih sedikit objek, tapi lebih yakin).",
    )
    match_persons = st.sidebar.toggle(
        "Cocokkan helm & vest dengan person",
        value=True,
        help="Helm/vest hanya dihitung jika berada di dalam seorang person, "
        "dan tiap person diberi status APD-nya. Matikan untuk melihat hasil mentah model.",
    )
    return Settings(mode, confidence, match_persons)


def show_summary(detections: list[Detection]) -> None:
    """Menampilkan jumlah Helm, Vest, dan Person (selalu tampil, 0 jika tidak ada)."""
    counts = count_by_class(detections)

    st.subheader("Jumlah terdeteksi")
    for column, (class_name, label) in zip(st.columns(3), SUMMARY_LABELS.items()):
        column.metric(label=label, value=counts[class_name])


# ---------------------------------------------------------------------------
# Mode 1: gambar
# ---------------------------------------------------------------------------
def pick_image() -> Image.Image | None:
    """Meminta pengguna memilih gambar: unggah sendiri atau gambar contoh."""
    source = st.radio("Sumber gambar", [SOURCE_UPLOAD, SOURCE_SAMPLE], horizontal=True)

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


def run_image_mode(settings: Settings) -> None:
    """Mode gambar: pilih gambar, deteksi, lalu tampilkan hasilnya."""
    image = pick_image()
    if image is None:
        return

    with st.spinner("Mendeteksi..."):
        detections = detect_objects(load_model(), image, settings, IMAGE_SIZE)

    original_column, result_column = st.columns(2)
    original_column.subheader("Gambar asli")
    original_column.image(image, use_container_width=True)
    result_column.subheader("Hasil deteksi")
    result_column.image(draw_detections(np.array(image), detections), use_container_width=True)

    show_summary(detections)


# ---------------------------------------------------------------------------
# Mode 2: video (folder videos, YouTube, atau file unggahan)
# ---------------------------------------------------------------------------
def run_video_mode(settings: Settings) -> None:
    """Mode video: pilih sumber video, lalu deteksi secara langsung."""
    target_fps = st.sidebar.slider(
        "Frame diproses per detik",
        min_value=1,
        max_value=30,
        value=10,
        help="Makin besar makin halus tapi makin berat. Video asli 24-30 fps, jadi "
        "30 berarti setiap frame diproses. Turunkan jika terasa lambat "
        "(3-5 sudah cukup untuk memantau APD).",
    )
    max_seconds = st.sidebar.slider(
        "Durasi video diproses (detik)",
        min_value=10,
        max_value=300,
        value=60,
        step=10,
    )

    source = pick_video()
    if not st.button("Mulai deteksi", type="primary"):
        st.info("Atur pengaturan di sidebar, lalu klik **Mulai deteksi**.")
        return
    if source.is_empty:
        st.warning("Isi link YouTube atau unggah file video terlebih dahulu.")
        return

    try:
        with open_video(source, YOUTUBE_RESOLUTION) as capture:
            if capture.isOpened():
                process_video(capture, settings, target_fps, max_seconds)
            else:
                st.error(
                    "Video tidak bisa dibuka (format mungkin tidak didukung). "
                    f"Coba video lain atau pilih '{UPLOAD_VIDEO}'."
                )
    except yt_dlp.utils.DownloadError:
        st.error(
            "Video YouTube tidak bisa diambil. Periksa link atau coba lagi nanti. "
            f"Di Streamlit Cloud YouTube sering memblokir server; pilih '{UPLOAD_VIDEO}' "
            "sebagai gantinya."
        )


def pick_video() -> VideoSource:
    """Meminta pengguna memilih video: contoh dari folder, link YouTube, atau unggahan."""
    sample_videos = {
        path.stem: path
        for path in sorted(VIDEOS_DIR.glob("*"))
        if path.suffix[1:].lower() in VIDEO_TYPES
    }
    choice = st.selectbox("Pilih video", [*sample_videos, CUSTOM_VIDEO, UPLOAD_VIDEO])

    if choice == UPLOAD_VIDEO:
        return VideoSource(upload=st.file_uploader("Unggah video", type=VIDEO_TYPES))
    if choice == CUSTOM_VIDEO:
        link = st.text_input("Link YouTube", placeholder="https://www.youtube.com/watch?v=...")
        return VideoSource(youtube_url=link)
    return VideoSource(file=sample_videos[choice])


def open_youtube(youtube_url: str, resolution: int, folder: str) -> cv2.VideoCapture:
    """Membuka video YouTube. Jika stream langsung gagal dibuka, unduh dulu ke `folder`."""
    with st.spinner("Mengambil video dari YouTube..."):
        capture = cv2.VideoCapture(get_stream_url(youtube_url, resolution))
        if capture.isOpened():
            return capture
        capture.release()

        # Di beberapa server (misalnya cloud) stream langsung tidak terbaca OpenCV,
        # sedangkan hasil unduhan yt-dlp biasanya terbaca.
        return cv2.VideoCapture(download_video(youtube_url, resolution, folder))


def open_capture(source: VideoSource, resolution: int, folder: str) -> cv2.VideoCapture:
    """Membuka video dari sumber yang dipilih. `folder` dipakai untuk file sementara."""
    if source.file is not None:
        return cv2.VideoCapture(str(source.file))

    if source.upload is not None:
        video_path = Path(folder) / Path(source.upload.name).name
        video_path.write_bytes(source.upload.getbuffer())
        return cv2.VideoCapture(str(video_path))

    return open_youtube(source.youtube_url, resolution, folder)


@contextmanager
def open_video(source: VideoSource, resolution: int):
    """Membuka video dan menutupnya otomatis. File sementara dihapus saat selesai."""
    with tempfile.TemporaryDirectory() as folder:
        capture = open_capture(source, resolution, folder)
        try:
            yield capture
        finally:
            capture.release()


def process_video(
    capture: cv2.VideoCapture, settings: Settings, target_fps: int, max_seconds: int
) -> None:
    """Mendeteksi frame demi frame dan menampilkannya secara langsung."""
    model = load_model()
    video_fps = capture.get(cv2.CAP_PROP_FPS) or FALLBACK_VIDEO_FPS
    step = max(1, round(video_fps / target_fps))  # ambil 1 dari setiap `step` frame
    effective_fps = video_fps / step  # FPS yang benar-benar diproses & ditampilkan
    seconds_per_frame = 1 / effective_fps

    # Jumlah frame yang akan diproses (video pendek bisa lebih kecil dari batas durasi).
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(max_seconds * effective_fps)
    total_frames = min(max_frames, video_frames // step) if video_frames > 0 else max_frames

    progress = st.progress(0.0)
    frame_box = st.empty()
    summary_box = st.empty()

    for number, frame in enumerate(sample_frames(capture, step, max_frames), start=1):
        started_at = time.perf_counter()

        detections = detect_objects(model, frame, settings, VIDEO_IMAGE_SIZE)
        frame_rgb = frame[..., ::-1]  # frame OpenCV berformat BGR -> RGB
        frame_box.image(
            draw_detections(frame_rgb, detections),
            use_container_width=True,
            output_format="JPEG",  # lebih kecil dikirim ke browser untuk frame beresolusi besar
        )
        with summary_box.container():
            show_summary(detections)
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

    settings = show_sidebar()

    if settings.mode == MODE_VIDEO:
        run_video_mode(settings)
    else:
        run_image_mode(settings)


main()
