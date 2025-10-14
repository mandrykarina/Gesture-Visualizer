import os
import pandas as pd
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from .extractor import GestureLandmarkExtractor
from .utils import load_annotations, ensure_dirs

class BatchProcessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        # Директория, где лежит processor.py
        base_dir = os.path.dirname(__file__)
        # Строим абсолютные пути, смещаясь из gesture_processing
        self.paths = {}
        for key, rel in cfg["paths"].items():
            self.paths[key] = os.path.abspath(os.path.join(base_dir, rel))
        # Создаём папки вывода
        ensure_dirs([self.paths["output_landmarks"], self.paths["output_visuals"]])
        # Инициализируем экстрактор
        self.extractor = GestureLandmarkExtractor(
            frame_interval_ms=cfg["frame_interval_ms"],
            min_detection_confidence=cfg["min_detection_confidence"],
            min_tracking_confidence=cfg["min_tracking_confidence"]
        )
        self.max_videos = cfg["max_videos"]
        self.num_workers = cfg["num_workers"]

    def run(self):
        raw_dir = self.paths["raw_videos"]
        print("RAW VIDEOS DIR:", raw_dir)
        if not os.path.isdir(raw_dir):
            print(f"❌ Папка raw_videos не найдена: {raw_dir}")
            return []

        existing = os.listdir(raw_dir)
        print(f"Найдено видеофайлов в папке: {len(existing)} -> {existing[:5]}")

        df = load_annotations(self.paths["annotations"])
        df = df.head(self.max_videos)
        print(f"Обрабатываю первые {len(df)} видео из аннотаций")

        tasks = []
        for _, row in df.iterrows():
            vid_name = f"{row['attachment_id']}.mp4"
            vid = os.path.join(raw_dir, vid_name)
            out = os.path.join(self.paths["output_landmarks"], f"{row['attachment_id']}.json")
            if os.path.exists(vid):
                tasks.append((vid, out, row["text"]))
            else:
                print(f"⚠️  Видео не найдено и пропускается: {vid_name}")

        print(f"Подготовлено задач: {len(tasks)}")
        if not tasks:
            print("❌ Нечего обрабатывать — проверьте пути и имена файлов")
            return []

        results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as exec:
            futures = [exec.submit(self.extractor.extract, vid, out, label)
                       for vid, out, label in tasks]
            for f in tqdm(as_completed(futures), total=len(futures)):
                res = f.result()
                if res:
                    results.append(res)

        print(f"Обработано файлов: {len(results)}/{len(tasks)}")
        return results

if __name__ == "__main__":
    # Загружаем конфиг
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    print("Использую конфиг:", cfg_path)
    cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
    processor = BatchProcessor(cfg)
    processor.run()
