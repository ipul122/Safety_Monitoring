"""Aplikasi Streamlit: deteksi APD (helm & rompi).

Dua mode (dipilih lewat dropdown "Mode" di sidebar kiri):
    1. Images : unggah foto atau pakai gambar contoh, lalu lihat hasil deteksinya.
    2. Video  : pilih video contoh (folder videos) atau unggah file video. Seluruh
                video diproses dulu, lalu hasilnya diputar mulus beserta suaranya.

Cara menjalankan:
    streamlit run app.py
"""

import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import streamlit as st
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
VIDEO_IMAGE_SIZE = 640  # ukuran masuk model untuk video; lebih kecil = lebih cepat

# Video contoh: semua file di folder ini muncul otomatis di dropdown.
VIDEOS_DIR = Path(__file__).parent / "videos"
UPLOAD_VIDEO = "Unggah file video"
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]
MAX_OUTPUT_SIDE = 1280  # video besar diperkecil agar proses & pemutaran ringan
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
    upload: object | None = None  # file video yang diunggah pengguna

    @property
    def is_empty(self) -> bool:
        return self.file is None and self.upload is None

    @property
    def key(self) -> str:
        """Penanda unik sumber, untuk tahu apakah hasil tersimpan masih cocok."""
        if self.file is not None:
            return f"file:{self.file.name}:{self.file.stat().st_mtime}"
        return f"upload:{self.upload.file_id}"


@dataclass(frozen=True)
class VideoResult:
    """Hasil pemrosesan satu video."""

    video: bytes  # file mp4 yang sudah diberi kotak deteksi
    is_portrait: bool
    has_audio: bool
    max_counts: Counter  # jumlah terbanyak tiap kelas dalam satu frame
    violations: dict[str, list[int]]  # {kelas pelanggaran: daftar detik}


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
# Memproses video
# ---------------------------------------------------------------------------
def get_video_path(source: VideoSource, folder: Path) -> Path:
    """Menyediakan path video. File unggahan disimpan dulu ke `folder`."""
    if source.file is not None:
        return source.file

    path = folder / f"unggahan{Path(source.upload.name).suffix}"
    path.write_bytes(source.upload.getbuffer())
    return path


def limit_size(frame: np.ndarray) -> np.ndarray:
    """Memperkecil frame jika sisi terpanjangnya melebihi MAX_OUTPUT_SIDE."""
    scale = MAX_OUTPUT_SIDE / max(frame.shape[:2])
    if scale >= 1:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def open_writer(path: Path, frame: np.ndarray, fps: float):
    """Membuka penulis video H.264 (format yang bisa diputar browser)."""
    height, width = frame.shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(path),
        size=(width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="bgr24",
        macro_block_size=2,  # H.264 butuh lebar & tinggi genap
        quality=None,
        output_params=["-preset", "veryfast", "-crf", "23"],
    )
    writer.send(None)  # wajib sebelum mengirim frame pertama
    return writer


def annotate_video(
    video_path: Path, output_path: Path, settings: Settings, detect_fps: int, max_seconds: int
) -> tuple[float, Counter, dict[str, list[int]], bool]:
    """Membuat video baru (tanpa suara) berisi kotak deteksi.

    Semua frame ditulis agar hasilnya mulus, tetapi deteksi hanya dijalankan
    `detect_fps` kali per detik; kotak terakhir dipakai untuk frame di antaranya.
    Mengembalikan (durasi detik, jumlah terbanyak, detik pelanggaran, video vertikal?).
    """
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or FALLBACK_VIDEO_FPS
    detect_every = max(1, round(fps / detect_fps))

    total_frames = int(max_seconds * fps)
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frames > 0:
        total_frames = min(total_frames, video_frames)

    model = load_model()
    max_counts: Counter = Counter()
    violations: dict[str, set[int]] = {bad: set() for _, bad, _ in EQUIPMENT_CHECKS}
    detections: list[Detection] = []
    writer = None
    frames_written = 0
    is_portrait = False
    progress = st.progress(0.0, text="Memproses video...")

    try:
        for index in range(total_frames):
            success, frame = capture.read()
            if not success:  # video habis
                break
            frame = limit_size(frame)

            if index % detect_every == 0:
                detections = detect_objects(model, frame, settings, VIDEO_IMAGE_SIZE)
                counts = count_by_class(detections)
                max_counts |= counts  # simpan nilai terbesar tiap kelas
                for name in violations.keys() & counts.keys():
                    violations[name].add(int(index / fps))

                done = (index + 1) / total_frames
                progress.progress(done, text=f"Memproses video... {done:.0%}")

            annotated = draw_detections(frame[..., ::-1], detections)
            annotated_bgr = np.ascontiguousarray(np.asarray(annotated)[..., ::-1])
            if writer is None:
                writer = open_writer(output_path, annotated_bgr, fps)
                is_portrait = frame.shape[0] > frame.shape[1]
            writer.send(annotated_bgr)
            frames_written += 1
    finally:
        capture.release()
        if writer is not None:
            writer.close()
        progress.empty()

    if frames_written == 0:
        raise ValueError("Video tidak bisa dibaca.")
    seconds = {name: sorted(values) for name, values in violations.items()}
    return frames_written / fps, max_counts, seconds, is_portrait


def add_audio(silent_path: Path, source_path: Path, output_path: Path, duration: float) -> bool:
    """Menggabungkan suara video asli ke video hasil. False jika gagal."""
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-i", str(silent_path),
        "-i", str(source_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",  # tanda tanya: tidak masalah jika video asli tanpa suara
        "-c:v", "copy",
        "-c:a", "aac",
        "-t", f"{duration:.3f}",
        "-movflags", "+faststart",  # agar browser bisa langsung memutar
        str(output_path),
    ]
    return subprocess.run(command, capture_output=True).returncode == 0


def render_video(
    source: VideoSource, settings: Settings, detect_fps: int, max_seconds: int
) -> VideoResult:
    """Memproses video dari awal sampai siap diputar. File sementara dihapus otomatis."""
    with tempfile.TemporaryDirectory() as temp_dir:
        folder = Path(temp_dir)
        video_path = get_video_path(source, folder)
        silent_path = folder / "tanpa_suara.mp4"
        final_path = folder / "hasil.mp4"

        duration, max_counts, violations, is_portrait = annotate_video(
            video_path, silent_path, settings, detect_fps, max_seconds
        )
        with st.spinner("Menambahkan suara..."):
            has_audio = add_audio(silent_path, video_path, final_path, duration)

        result_path = final_path if has_audio else silent_path
        return VideoResult(
            result_path.read_bytes(), is_portrait, has_audio, max_counts, violations
        )


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


def show_summary(counts: Counter, title: str = "Jumlah terdeteksi") -> None:
    """Menampilkan jumlah Helm, Vest, dan Person (selalu tampil, 0 jika tidak ada)."""
    st.subheader(title)
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

    show_summary(count_by_class(detections))


# ---------------------------------------------------------------------------
# Mode 2: video (folder videos atau file unggahan)
# ---------------------------------------------------------------------------
def pick_video() -> VideoSource:
    """Meminta pengguna memilih video: contoh dari folder atau unggahan sendiri."""
    sample_videos = {
        path.stem: path
        for path in sorted(VIDEOS_DIR.glob("*"))
        if path.suffix[1:].lower() in VIDEO_TYPES
    }
    choice = st.selectbox("Pilih video", [*sample_videos, UPLOAD_VIDEO])

    if choice == UPLOAD_VIDEO:
        return VideoSource(upload=st.file_uploader("Unggah video", type=VIDEO_TYPES))
    return VideoSource(file=sample_videos[choice])


def show_video_result(result: VideoResult) -> None:
    """Memutar video hasil, tombol unduh, dan ringkasan deteksi."""
    # Video vertikal ditaruh di kolom tengah agar tidak terlalu tinggi.
    player = st.columns([1, 1, 1])[1] if result.is_portrait else st.container()
    player.video(result.video, format="video/mp4", autoplay=True)
    if not result.has_audio:
        st.warning("Suara tidak bisa ditambahkan; video diputar tanpa suara.")
    st.download_button("Unduh hasil", result.video, "hasil_deteksi.mp4", "video/mp4")

    show_summary(result.max_counts, "Jumlah terbanyak dalam satu frame")
    for _, bad, word in EQUIPMENT_CHECKS:
        seconds = result.violations[bad]
        if seconds:
            st.error(f"Tanpa {word} terdeteksi pada detik: {', '.join(map(str, seconds))}")


def run_video_mode(settings: Settings) -> None:
    """Mode video: pilih video, proses seluruhnya, lalu putar hasilnya."""
    detect_fps = st.sidebar.slider(
        "Frame diproses per detik",
        min_value=1,
        max_value=30,
        value=5,
        help="Berapa kali deteksi dijalankan per detik video. Video hasil tetap mulus. "
        "Makin besar, kotak makin akurat mengikuti gerakan tetapi proses makin lama.",
    )
    max_seconds = st.sidebar.slider(
        "Durasi video diproses (detik)",
        min_value=10,
        max_value=300,
        value=30,
        step=10,
        help="Dihitung dari awal video. Makin panjang, makin lama menunggu.",
    )

    source = pick_video()
    # Hasil disimpan per kombinasi pilihan, jadi tidak diproses ulang saat halaman dimuat ulang.
    request = None
    if not source.is_empty:
        # Simpan nilai biasa (bukan objek Settings): kelas dibuat ulang tiap rerun Streamlit.
        request = (
            source.key, settings.confidence, settings.match_persons, detect_fps, max_seconds
        )

    if st.button("Proses video", type="primary"):
        if source.is_empty:
            st.warning("Unggah file video terlebih dahulu.")
            return
        try:
            result = render_video(source, settings, detect_fps, max_seconds)
        except ValueError:
            st.error("Video tidak bisa dibaca. Coba file video lain.")
            return
        st.session_state["video_result"] = (request, result)

    saved = st.session_state.get("video_result")
    if request is None or saved is None or saved[0] != request:
        st.info("Klik **Proses video**. Hasilnya diputar setelah proses selesai.")
        return
    show_video_result(saved[1])


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
