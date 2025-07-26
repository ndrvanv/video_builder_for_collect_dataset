import os, json, csv, hashlib, logging, time, shutil, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

import cv2
import imagehash
from PIL import Image
import yt_dlp
import requests
import random

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("dataset_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка конфигурации с проверкой
try:
    with open("config.json", "r", encoding="utf-8") as f:
        CFG = json.load(f)
except FileNotFoundError:
    logger.error("config.json не найден!")
    raise
except json.JSONDecodeError as e:
    logger.error(f"Ошибка в config.json: {e}")
    raise

OUT_DIR = Path("dataset")
INDEX_CSV = OUT_DIR / "index.csv"
TMP_DIR = Path("_tmp")
TMP_DIR.mkdir(exist_ok=True)



# 1. Поиск видео

def search_yt(keyword: str, limit: int) -> List[str]:
    """Поиск видео на YouTube по ключевому слову"""
    try:
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "socket_timeout": 30,
            "retries": 3
        }
        query = f"ytsearch{limit}:{keyword}"

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            urls = []
            for entry in info.get("entries", []):
                if entry and entry.get("webpage_url"):
                    urls.append(entry["webpage_url"])

            logger.info(f"YouTube: найдено {len(urls)} видео для '{keyword}'")
            return urls

    except Exception as e:
        logger.error(f"Ошибка поиска YouTube для '{keyword}': {e}")
        return []


def search_pexels(keyword: str, limit: int) -> List[str]:
    """Поиск видео на Pexels по ключевому слову"""
    try:
        url = "https://api.pexels.com/videos/search"
        headers = {"Authorization": CFG["pexels_key"]}
        params = {"query": keyword, "per_page": min(limit, 80)}

        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        urls = []

        for video in data.get("videos", []):
            video_files = video.get("video_files", [])
            if video_files:
                # Выбираем файл с наилучшим качеством <= 720p
                best_file = None
                for vf in video_files:
                    if vf.get("height", 0) <= 720:
                        if not best_file or vf.get("height", 0) > best_file.get("height", 0):
                            best_file = vf

                if best_file and best_file.get("link"):
                    urls.append(best_file["link"])

        logger.info(f"Pexels: найдено {len(urls)} видео для '{keyword}'")
        return urls

    except Exception as e:
        logger.error(f"Ошибка поиска Pexels для '{keyword}': {e}")
        return []



# 2. Утилиты для работы с файлами

def md5_file(path: Path) -> str:
    """Вычисление MD5 хеша файла"""
    hash_md5 = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Ошибка вычисления MD5 для {path}: {e}")
        return ""


def probe_duration(path: Path) -> float:
    """Получение длительности видео через ffprobe"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return float(result.stdout.strip())
        else:
            logger.error(f"ffprobe ошибка для {path}: {result.stderr}")
            return 0.0
    except Exception as e:
        logger.error(f"Ошибка определения длительности {path}: {e}")
        return 0.0


def check_ffmpeg_installed():
    """Проверка установки ffmpeg и ffprobe"""
    for tool in ["ffmpeg", "ffprobe"]:
        try:
            subprocess.run([tool, "-version"], capture_output=True, timeout=10)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error(f"{tool} не установлен или недоступен!")
            return False
    return True



# 3. Скачивание и обрезка

def download_and_trim(url: str, keyword: str, seen_hashes: set) -> Tuple[Optional[Path], Optional[dict]]:
    """Скачивание видео и создание клипа"""
    try:
        # Генерируем уникальный ID для файла
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        raw_path = TMP_DIR / f"{keyword}_{url_hash}.mp4"

        # Настройки для yt-dlp
        ydl_opts = {
            "outtmpl": str(raw_path),
            "format": "best[height<=720][ext=mp4]/best[height<=720]/best",
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 60,
            "retries": 3,
            "fragment_retries": 3,
            "file_access_retries": 3,
            "force_ipv4": True,
        }

        # Попытка скачивания
        logger.info(f"Скачиваем: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Проверяем, что файл скачался
        if not raw_path.exists():
            logger.warning(f"Файл не скачался: {url}")
            return None, None

        # Проверяем длительность
        duration = probe_duration(raw_path)
        if duration < CFG["min_duration"]:
            logger.info(f"Видео слишком короткое ({duration}s): {url}")
            raw_path.unlink(missing_ok=True)
            return None, None

        # Выбираем случайный момент для обрезки
        clip_duration = CFG["clip_duration"]
        max_start = max(0, duration - clip_duration)
        start_time = random.uniform(0, max_start) if max_start > 0 else 0

        # Создаем директорию для ключевого слова
        keyword_dir = OUT_DIR / keyword
        keyword_dir.mkdir(parents=True, exist_ok=True)

        final_path = keyword_dir / f"{url_hash}.mp4"

        # Обрезаем видео
        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", str(start_time),
            "-i", str(raw_path),
            "-t", str(clip_duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-avoid_negative_ts", "make_zero",
            "-y", str(final_path)
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=120)

        # Удаляем временный файл
        raw_path.unlink(missing_ok=True)

        if result.returncode != 0:
            logger.error(f"Ошибка ffmpeg: {result.stderr.decode()}")
            return None, None

        # Проверяем, что итоговый файл создался
        if not final_path.exists() or final_path.stat().st_size == 0:
            logger.warning(f"Итоговый файл пуст или не создался: {final_path}")
            final_path.unlink(missing_ok=True)
            return None, None

        # Проверяем на дубликаты
        file_hash = md5_file(final_path)
        if not file_hash or file_hash in seen_hashes:
            logger.info(f"Дубликат или ошибка хеша: {final_path}")
            final_path.unlink(missing_ok=True)
            return None, None

        # Метаданные
        metadata = {
            "path": str(final_path.relative_to(OUT_DIR)),
            "keyword": keyword,
            "hash": file_hash,
            "url": url,
            "duration": duration,
            "clip_start": start_time
        }

        logger.info(f"Успешно создан клип: {final_path}")
        return final_path, metadata

    except Exception as e:
        logger.error(f"Ошибка обработки {url}: {e}")
        # Очистка временных файлов
        for temp_file in [raw_path] if 'raw_path' in locals() else []:
            temp_file.unlink(missing_ok=True)
        return None, None



# 4. Детекция водяных знаков

def has_watermark(video_path: Path, threshold: int = 2, check_duration: float = 1.0) -> bool:
    """
    Проверка на наличие статичных водяных знаков в углах видео
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Не удалось открыть видео: {video_path}")
            return False

        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False

        h, w = frame.shape[:2]

        # Определяем области для проверки (углы)
        roi_size_h, roi_size_w = int(h * 0.15), int(w * 0.15)

        def extract_rois(frame):
            return [
                frame[0:roi_size_h, 0:roi_size_w],  # левый верхний
                frame[0:roi_size_h, -roi_size_w:],  # правый верхний
                frame[-roi_size_h:, 0:roi_size_w],  # левый нижний
                frame[-roi_size_h:, -roi_size_w:]  # правый нижний
            ]

        # Хеши первого кадра
        initial_rois = extract_rois(frame)
        prev_hashes = []

        for roi in initial_rois:
            try:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                pil_roi = Image.fromarray(gray_roi)
                hash_val = imagehash.average_hash(pil_roi)
                prev_hashes.append(hash_val)
            except Exception as e:
                logger.warning(f"Ошибка создания хеша: {e}")
                cap.release()
                return False

        # Счетчики идентичных кадров для каждого угла
        identical_counts = [0] * 4

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frames_to_check = int(fps * check_duration)
        frame_count = 0

        while frame_count < frames_to_check:
            ret, frame = cap.read()
            if not ret:
                break

            current_rois = extract_rois(frame)

            for i, (roi, prev_hash) in enumerate(zip(current_rois, prev_hashes)):
                try:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    pil_roi = Image.fromarray(gray_roi)
                    current_hash = imagehash.average_hash(pil_roi)

                    # Сравниваем хеши
                    if abs(prev_hash - current_hash) <= threshold:
                        identical_counts[i] += 1
                    else:
                        identical_counts[i] = 0

                    # Если в одном углу слишком много идентичных кадров - это водяной знак
                    if identical_counts[i] > frames_to_check * 0.8:
                        cap.release()
                        return True

                except Exception:
                    continue

            frame_count += 1

        cap.release()
        return False

    except Exception as e:
        logger.error(f"Ошибка детекции водяного знака в {video_path}: {e}")
        return False


###############################################################################
# 5. Главная функция
###############################################################################
def main():
    """Основная функция сборки датасета"""
    logger.info("Запуск сборки видео-датасета")

    # Проверяем наличие необходимых инструментов
    if not check_ffmpeg_installed():
        logger.error("ffmpeg/ffprobe не установлены. Установите их для продолжения.")
        return

    # Создаем директории
    OUT_DIR.mkdir(exist_ok=True)

    # Загружаем уже обработанные хеши
    seen_hashes = set()
    csv_exists = INDEX_CSV.exists() and INDEX_CSV.stat().st_size > 0

    if csv_exists:
        try:
            with open(INDEX_CSV, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                seen_hashes = {row["hash"] for row in reader if row.get("hash")}
            logger.info(f"Загружено {len(seen_hashes)} уже обработанных видео")
        except Exception as e:
            logger.error(f"Ошибка чтения существующего индекса: {e}")

    # Открываем CSV для записи
    try:
        with open(INDEX_CSV, "a", newline="", encoding="utf-8") as csv_file:
            fieldnames = ["path", "keyword", "hash", "url", "duration", "clip_start"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # Записываем заголовок если файл новый
            if not csv_exists:
                writer.writeheader()

            # Обрабатываем каждое ключевое слово
            for keyword in CFG["keywords"]:
                logger.info(f"Обработка ключевого слова: '{keyword}'")

                # Собираем URL из разных источников
                all_urls = []

                # YouTube
                yt_urls = search_yt(keyword, CFG["max_per_kw"])
                all_urls.extend(yt_urls)

                # Pexels
                pexels_urls = search_pexels(keyword, CFG["max_per_kw"])
                all_urls.extend(pexels_urls)

                # Перемешиваем для разнообразия
                random.shuffle(all_urls)

                logger.info(f"Найдено {len(all_urls)} URL для '{keyword}'")

                if not all_urls:
                    logger.warning(f"Не найдено видео для '{keyword}'")
                    continue

                # Обрабатываем видео в многопоточном режиме
                successful_downloads = 0

                with ThreadPoolExecutor(max_workers=CFG["workers"]) as executor:
                    # Запускаем задачи
                    future_to_url = {
                        executor.submit(download_and_trim, url, keyword, seen_hashes): url
                        for url in all_urls
                    }

                    # Обрабатываем результаты
                    for future in as_completed(future_to_url):
                        url = future_to_url[future]
                        try:
                            video_path, metadata = future.result()

                            if video_path and metadata:
                                # Проверяем на водяные знаки
                                if has_watermark(video_path):
                                    logger.info(f"Обнаружен водяной знак, удаляем: {video_path}")
                                    video_path.unlink(missing_ok=True)
                                    continue

                                # Записываем в CSV
                                writer.writerow(metadata)
                                csv_file.flush()  # Сразу сохраняем на диск

                                # Добавляем в множество обработанных
                                seen_hashes.add(metadata["hash"])
                                successful_downloads += 1

                                logger.info(f"✓ Добавлен клип: {metadata['path']}")

                        except Exception as e:
                            logger.error(f"Ошибка обработки {url}: {e}")

                logger.info(f"Завершена обработка '{keyword}': {successful_downloads} успешных клипов")

                # Небольшая пауза между ключевыми словами
                if keyword != CFG["keywords"][-1]:  # не последнее слово
                    time.sleep(5)

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        raise

    finally:
        # Очистка временных файлов
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR, ignore_errors=True)

        logger.info("Сборка датасета завершена")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        raise