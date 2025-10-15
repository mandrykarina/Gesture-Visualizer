import os
import json
import pandas as pd
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from .utils import load_annotations, ensure_dirs


def _process_video(vid, out, label, cfg):
    """Извлекает точки из одного видео (выполняется в отдельном процессе)."""
    from .extractor import GestureLandmarkExtractor
    extractor = GestureLandmarkExtractor(
        frame_interval_ms=cfg["frame_interval_ms"],
        min_detection_confidence=cfg["min_detection_confidence"],
        min_tracking_confidence=cfg["min_tracking_confidence"]
    )
    return extractor.extract(vid, out, label)


class BatchProcessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        base_dir = os.path.dirname(__file__)

        # абсолютные пути
        self.paths = {
            key: os.path.abspath(os.path.join(base_dir, rel))
            for key, rel in cfg["paths"].items()
        }

        ensure_dirs([self.paths["output_landmarks"], self.paths["output_visuals"]])
        self.max_videos = cfg["max_videos"]
        self.num_workers = cfg["num_workers"]

    def run(self):
        raw_dir = self.paths["raw_videos"]
        ann_path = self.paths["annotations"]

        print(f"📂 RAW VIDEOS DIR: {raw_dir}")
        if not os.path.isdir(raw_dir):
            print(f"❌ Папка raw_videos не найдена: {raw_dir}")
            return []

        df = load_annotations(ann_path)
        existing_videos = {os.path.splitext(f)[0] for f in os.listdir(raw_dir)}
        df = df[df["attachment_id"].isin(existing_videos)]

        print(f"[INFO] Найдено совпадений видео/аннотаций: {len(df)}")
        if len(df) == 0:
            print("❌ Нет совпадений — проверь имена файлов и CSV")
            return []

        print(f"[INFO] Пример строк аннотаций:")
        print(df.head(3))

        # Ограничим, если нужно
        if self.max_videos and len(df) > self.max_videos:
            df = df.head(self.max_videos)
            print(f"[INFO] Обрабатываю первые {len(df)} совпадений")

        # Подготавливаем задачи
        tasks = []
        for _, row in df.iterrows():
            vid_name = f"{row['attachment_id']}.mp4"
            vid = os.path.join(raw_dir, vid_name)
            out = os.path.join(self.paths["output_landmarks"], f"{row['attachment_id']}.json")
            tasks.append((vid, out, row["text"]))

        print(f"[INFO] Подготовлено задач: {len(tasks)}")

        # Обработка видео
        results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as exec:
            futures = [exec.submit(_process_video, vid, out, label, self.cfg)
                       for vid, out, label in tasks]

            for f in tqdm(as_completed(futures), total=len(futures), desc="Обработка видео"):
                try:
                    res = f.result()
                    if res:
                        results.append(res)
                except Exception as e:
                    print(f"⚠️ Ошибка при обработке видео: {e}")

        print(f"✅ Обработано файлов: {len(results)}/{len(tasks)}")

        # Формируем общий словарь: слово -> список json путей
        gesture_dict = {}
        for json_path in results:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                gesture = data["gesture"]
                gesture_dict.setdefault(gesture, []).append(data["frames"])

        dataset_path = os.path.join(self.paths["output_landmarks"], "gesture_dataset.json")
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(gesture_dict, f, ensure_ascii=False, indent=2)

        print(f"💾 Итоговый словарь сохранён: {dataset_path}")
        return gesture_dict


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processor = BatchProcessor(cfg)
    processor.run()
