import os
import json
import pandas as pd
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from .utils import load_annotations, ensure_dirs


def _process_video(vid, out, label, cfg):
    """Выполняется в дочернем процессе. Создаёт свой экземпляр медиапайпа."""
    try:
        from .extractor import GestureLandmarkExtractor
    except Exception as e:
        print(f"[ERROR worker import] {e}")
        return None

    try:
        extractor = GestureLandmarkExtractor(
            frame_interval_ms=cfg.get("frame_interval_ms", 100),
            min_detection_confidence=cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=cfg.get("min_tracking_confidence", 0.5)
        )
        return extractor.extract(vid, out, label)
    except Exception as e:
        print(f"[ERROR worker] Exception processing {vid}: {e}")
        return None


class BatchProcessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        base_dir = os.path.dirname(__file__)
        self.paths = {
            key: os.path.abspath(os.path.join(base_dir, rel))
            for key, rel in cfg["paths"].items()
        }

        ensure_dirs([self.paths["output_landmarks"], self.paths["output_visuals"]])
        self.max_videos = cfg.get("max_videos", 0) or 0
        self.num_workers = max(1, int(cfg.get("num_workers", 1)))
        try:
            import multiprocessing
            self.num_workers = min(self.num_workers, max(1, multiprocessing.cpu_count()))
        except Exception:
            pass
        self.batch_size = int(cfg.get("batch_size", 50))

    def run(self):
        raw_dir = self.paths["raw_videos"]
        ann_path = self.paths["annotations"]

        print(f"📂 RAW VIDEOS DIR: {raw_dir}")
        if not os.path.isdir(raw_dir):
            print(f"❌ Папка raw_videos не найдена: {raw_dir}")
            return {}

        df_all = load_annotations(ann_path)
        existing_videos = {os.path.splitext(f)[0] for f in os.listdir(raw_dir)}
        df = df_all[df_all["attachment_id"].isin(existing_videos)].copy()

        print(f"[INFO] Найдено совпадений видео/аннотаций: {len(df)}")
        if len(df) == 0:
            print("❌ Нет совпадений — проверь имена файлов и CSV")
            return {}

        if self.max_videos and len(df) > self.max_videos:
            df = df.head(self.max_videos)
            print(f"[INFO] Обрабатываю первые {len(df)} совпадений (max_videos)")

        # --- формируем задачи (видео, путь json, слово) ---
        tasks = [
            (os.path.join(raw_dir, f"{row['attachment_id']}.mp4"),
             os.path.join(self.paths["output_landmarks"], f"{row['attachment_id']}.json"),
             str(row["text"]).strip().lower())
            for _, row in df.iterrows()
        ]

        # --- фильтрация по уникальным словам + проверка существующего файла ---
        seen_words = set()
        filtered_tasks = []
        for vid, out, label in tasks:
            if label not in seen_words and not os.path.exists(out):
                seen_words.add(label)
                filtered_tasks.append((vid, out, label))
        skipped = len(tasks) - len(filtered_tasks)

        tasks = filtered_tasks
        print(f"[INFO] Пропущено (повторы слов или уже обработанные): {skipped}")
        print(f"[INFO] К обработке уникальных слов: {len(tasks)}")

        processed_jsons = []

        # --- обработка батчами ---
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + self.batch_size]
            print(f"[INFO] Обработка батча {i//self.batch_size + 1} — задач: {len(batch)}")

            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(_process_video, vid, out, label, self.cfg) for vid, out, label in batch]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Батч {i//self.batch_size + 1}"):
                    try:
                        res = fut.result(timeout=None)
                        if res:
                            processed_jsons.append(res)
                    except Exception as e:
                        print(f"[ERROR] Ошибка в батче: {e}")

            print(f"[INFO] Батч {i//self.batch_size + 1} завершён. Промежуточно обработано: {len(processed_jsons)}")

        print(f"[INFO] Всего успешно обработано json: {len(processed_jsons)}")

        # --- сборка итогового словаря ---
        label_map = {}
        for p in processed_jsons:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                lbl = meta.get("gesture", "unknown")
                label_map.setdefault(lbl, []).append(p)
            except Exception as e:
                print(f"[WARN] не удалось открыть {p}: {e}")

        dataset_path = os.path.join(self.paths["output_landmarks"], "gesture_dataset.json")
        print(f"[INFO] Запись итогового словаря → {dataset_path}")

        try:
            labels = list(label_map.items())
            with open(dataset_path, "w", encoding="utf-8") as out:
                out.write("{\n")
                for idx, (label, paths) in enumerate(labels):
                    out.write(json.dumps(label, ensure_ascii=False))
                    out.write(": [")
                    first_example = True
                    for p in paths:
                        try:
                            with open(p, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                frames = data.get("frames", [])
                                if not first_example:
                                    out.write(",")
                                out.write(json.dumps(frames, ensure_ascii=False))
                                first_example = False
                        except Exception as e:
                            print(f"[WARN] при добавлении {p} в итог: {e}")
                    out.write("]")
                    if idx != len(labels) - 1:
                        out.write(",\n")
                out.write("\n}\n")
            print(f"✅ Итоговый словарь сохранён: {dataset_path}")
        except Exception as e:
            print(f"[ERROR] Не удалось записать итоговый словарь: {e}")

        return label_map


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processor = BatchProcessor(cfg)
    processor.run()
