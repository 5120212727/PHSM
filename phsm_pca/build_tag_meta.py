# build_tag_meta.py
import os
import re
import json
import pickle
from typing import Dict, List, Tuple


def normalize_tag_text(tag_text: str) -> str:
    """
    清洗 hashtag 文本，生成后续可用于文本编码的版本。
    例如:
        '#dogsofinstagram' -> 'dogsofinstagram'
        '#national_watermelon_day' -> 'national watermelon day'
        'Dog-Lover' -> 'dog lover'
    """
    text = tag_text.strip()

    # 去掉开头的 #
    if text.startswith("#"):
        text = text[1:]

    # 小写
    text = text.lower()

    # 下划线、横杠替换为空格
    text = text.replace("_", " ").replace("-", " ")

    # 去掉多余空白
    text = re.sub(r"\s+", " ", text).strip()

    return text


def parse_tag_line(line: str) -> Tuple[int, str]:
    """
    解析 tags 文件中的一行。
    兼容几种常见格式：
        0    dog
        0 dog
        0\t#dog
        147
    如果某一行只有 tag_id，没有标签文本，则用 tag_147 作为占位文本。
    """
    line = line.strip()
    if not line:
        raise ValueError("Empty line")

    # 优先按 tab 分
    parts = line.split("\t")
    if len(parts) >= 2:
        tag_id = int(parts[0].strip())
        tag_text = "\t".join(parts[1:]).strip()
        if tag_text == "":
            tag_text = f"tag_{tag_id}"
        return tag_id, tag_text

    # 再按空格分
    parts = line.split(maxsplit=1)
    if len(parts) == 2:
        tag_id = int(parts[0].strip())
        tag_text = parts[1].strip()
        if tag_text == "":
            tag_text = f"tag_{tag_id}"
        return tag_id, tag_text

    # 如果只有一个数字，比如 "147"
    if len(parts) == 1 and parts[0].isdigit():
        tag_id = int(parts[0])
        tag_text = f"tag_{tag_id}"
        return tag_id, tag_text

    raise ValueError(f"Cannot parse line: {line}")

def load_tags_file(tags_path: str) -> Dict[int, str]:
    """
    读取 tags 文件，返回:
        tag_id_to_raw_text
    """
    tag_id_to_raw_text: Dict[int, str] = {}

    with open(tags_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                tag_id, tag_text = parse_tag_line(line)
                tag_id_to_raw_text[tag_id] = tag_text
            except Exception as e:
                print(f"[WARN] line {line_no} parse failed: {e}")

    return tag_id_to_raw_text


def build_tag_meta(tag_id_to_raw_text: Dict[int, str]) -> Dict:
    """
    构建标签元数据。
    """
    tag_id_to_clean_text: Dict[int, str] = {}
    clean_text_to_tag_ids: Dict[str, List[int]] = {}

    for tag_id, raw_text in tag_id_to_raw_text.items():
        clean_text = normalize_tag_text(raw_text)
        tag_id_to_clean_text[tag_id] = clean_text

        if clean_text not in clean_text_to_tag_ids:
            clean_text_to_tag_ids[clean_text] = []
        clean_text_to_tag_ids[clean_text].append(tag_id)

    tag_ids_sorted = sorted(tag_id_to_raw_text.keys())

    meta = {
        "num_tags": len(tag_id_to_raw_text),
        "tag_ids": tag_ids_sorted,
        "tag_id_to_raw_text": tag_id_to_raw_text,
        "tag_id_to_clean_text": tag_id_to_clean_text,
        "clean_text_to_tag_ids": clean_text_to_tag_ids,
    }
    return meta


def save_outputs(meta: Dict, out_dir: str) -> None:
    """
    保存 pkl 和 json。
    """
    pkl_path = os.path.join(out_dir, "tag_meta.pkl")
    json_path = os.path.join(out_dir, "tag_meta.json")

    with open(pkl_path, "wb") as f:
        pickle.dump(meta, f)

    # json 里把 int key 转成 str，避免序列化问题
    json_meta = {
        "num_tags": meta["num_tags"],
        "tag_ids": meta["tag_ids"],
        "tag_id_to_raw_text": {str(k): v for k, v in meta["tag_id_to_raw_text"].items()},
        "tag_id_to_clean_text": {str(k): v for k, v in meta["tag_id_to_clean_text"].items()},
        "clean_text_to_tag_ids": {k: v for k, v in meta["clean_text_to_tag_ids"].items()},
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {pkl_path}")
    print(f"[OK] saved: {json_path}")


def main():
    root_dir = os.getcwd()
    tags_path = os.path.join(root_dir, "data", "tags")

    if not os.path.exists(tags_path):
        raise FileNotFoundError(f"Cannot find tags file: {tags_path}")

    print(f"[INFO] loading tags from: {tags_path}")
    tag_id_to_raw_text = load_tags_file(tags_path)

    if len(tag_id_to_raw_text) == 0:
        raise RuntimeError("No tags loaded. Please check the format of data/tags")

    meta = build_tag_meta(tag_id_to_raw_text)
    save_outputs(meta, root_dir)

    print(f"[INFO] total tags: {meta['num_tags']}")
    print("[INFO] first 10 examples:")
    for tag_id in meta["tag_ids"][:10]:
        raw_text = meta["tag_id_to_raw_text"][tag_id]
        clean_text = meta["tag_id_to_clean_text"][tag_id]
        print(f"  tag_id={tag_id} | raw='{raw_text}' | clean='{clean_text}'")


if __name__ == "__main__":
    main()