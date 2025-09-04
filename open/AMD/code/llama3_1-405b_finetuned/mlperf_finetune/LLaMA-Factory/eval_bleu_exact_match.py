# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import time

import fire
from datasets import load_dataset


try:
    import jieba  # type: ignore
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore
    from rouge_chinese import Rouge  # type: ignore

    jieba.setLogLevel(logging.CRITICAL)
    jieba.initialize()
except ImportError:
    print("Please install llamafactory with `pip install -e .[metrics]`.")
    raise

import re
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_chinese import Rouge

def extract_uuids(text: str):
    return re.findall(r'[\w]{8}-[\w]{4}-[\w]{4}-[\w]{4}-[\w]{12}', text.lower())

def compute_fuzzy_uuid_match_score(label, pred):
    label_uuids = sorted(extract_uuids(label))
    pred_uuids = sorted(extract_uuids(pred))
    if len(pred_uuids) == 0:
        return 0.0
    score = sum([
        pred==ref
        for pred, ref in zip(pred_uuids, label_uuids)
    ]) / len(pred_uuids) * 100
    return score

def compute_metrics(sample):
    pred = sample["predict"]
    label = sample["label"]

    # === BLEU & ROUGE ===
    hypothesis = list(jieba.cut(pred))
    reference = list(jieba.cut(label))

    bleu_score = sentence_bleu(
        [list(label)],
        list(pred),
        smoothing_function=SmoothingFunction().method3,
    )

    if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
        result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    else:
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        result = scores[0]

    metric_result = {}
    for k, v in result.items():
        metric_result[k] = round(v["f"] * 100, 4)

    metric_result["bleu-4"] = round(bleu_score * 100, 4)

    # === UUID 模糊匹配指标 ===
    pred_uuids = extract_uuids(pred)
    label_uuids = extract_uuids(label)

    pred_set = set([u.lower() for u in pred_uuids])
    label_set = set([u.lower() for u in label_uuids])

    if not pred_uuids:
        metric_result["uuid_precision"] = 0.0
        metric_result["uuid_recall"] = 0.0
        metric_result["uuid_f1"] = 0.0
        metric_result["uuid_jaccard"] = 0.0
        metric_result["uuid_exact_match"] = 0.0
        return metric_result

    matched_ref = sum([1.0 if ref in pred.lower() else 0.0 for ref in label_uuids])
    recall = matched_ref / len(label_uuids) if label_uuids else 0.0

    matched_pred = sum([1.0 if p in label.lower() else 0.0 for p in pred_uuids])
    precision = matched_pred / len(pred_uuids)

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    jaccard = len(pred_set & label_set) / len(pred_set | label_set) if (pred_set | label_set) else 0.0

    metric_result["uuid_precision"] = round(precision * 100, 4)
    metric_result["uuid_recall"] = round(recall * 100, 4)
    metric_result["uuid_f1"] = round(f1 * 100, 4)
    metric_result["uuid_jaccard"] = round(jaccard * 100, 4)

    # ✅ 模糊打分方式的 exact_match 替代（RULER 风格）
    metric_result["uuid_exact_match"] = compute_fuzzy_uuid_match_score(label, pred)

    return metric_result






def main(filename: str):
    start_time = time.time()
    dataset = load_dataset("json", data_files=filename, split="train")
    dataset = dataset.map(compute_metrics, num_proc=8, remove_columns=dataset.column_names)
    score_dict = dataset.to_dict()

    average_score = {}
    for task, scores in sorted(score_dict.items(), key=lambda x: x[0]):
        print(f"{task}: {sum(scores) / len(scores):.4f}")
        average_score[task] = sum(scores) / len(scores)

    with open("predictions_score.json", "w", encoding="utf-8") as f:
        json.dump(average_score, f, indent=4)

    print(f"\nDone in {time.time() - start_time:.3f}s.\nScore file saved to predictions_score.json")


if __name__ == "__main__":
    fire.Fire(main)

