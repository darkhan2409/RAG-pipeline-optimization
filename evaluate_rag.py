"""
evaluate_rag.py — Модуль для запуска экспериментов и RAGAS-оценки RAG pipeline.

Использование:
    python evaluate_rag.py                          # все 6 групп экспериментов
    python evaluate_rag.py --experiments 0 1        # только baseline и chunk_size
    python evaluate_rag.py --quick                  # быстрый тест (3 вопроса)
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict

import pandas as pd

# Автозагрузка .env если есть
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Lazy imports (тяжёлые зависимости) ────────────────────────────────────────

def _import_ragas():
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._context_recall import LLMContextRecall
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    return (
        EvaluationDataset, evaluate,
        Faithfulness, ResponseRelevancy, LLMContextRecall, LLMContextPrecisionWithReference,
        LangchainLLMWrapper, ChatOpenAI,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Конфигурация экспериментов ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    name: str
    method: str = "recursive"
    chunk_size: int = 1024
    chunk_overlap: int = 200
    top_k: int = 5
    rrf_alpha: float = 0.5
    use_reranking: bool = True
    collection_name: str | None = None
    rerank_threshold: float = 0.01


# ══════════════════════════════════════════════════════════════════════════════
# ── Golden Dataset ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def load_golden_dataset(path: str = "golden_dataset.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[Golden] Загружено {len(data)} вопросов из {path}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# ── Запуск эксперимента ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(cfg: ExperimentConfig, golden: list[dict]) -> dict:
    """
    Прогоняет golden dataset через RAG pipeline с заданной конфигурацией.
    Возвращает dict с config, samples, и RAGAS-метриками.
    """
    from ask_llama import ask_with_contexts, _bm25_cache

    # Сбрасываем BM25-кеш при смене коллекции/метода
    _bm25_cache.clear()

    samples = []
    total = len(golden)

    print(f"\n{'='*60}")
    print(f"  Эксперимент: {cfg.name}")
    print(f"  method={cfg.method}, chunk={cfg.chunk_size}/{cfg.chunk_overlap}, "
          f"top_k={cfg.top_k}, alpha={cfg.rrf_alpha}, rerank={cfg.use_reranking}")
    print(f"  Вопросов: {total}")
    print(f"{'='*60}")

    for i, item in enumerate(golden):
        q = item["question"]
        gt = item["ground_truth"]

        t0 = time.time()
        try:
            answer, contexts = ask_with_contexts(
                query=q,
                method=cfg.method,
                top_k=cfg.top_k,
                use_reranking=cfg.use_reranking,
                rrf_alpha=cfg.rrf_alpha,
                collection_name=cfg.collection_name,
                rerank_threshold=cfg.rerank_threshold,
            )
        except Exception as e:
            print(f"  [{i+1}/{total}] ОШИБКА: {e}")
            answer = f"Ошибка: {e}"
            contexts = []
        elapsed = time.time() - t0

        samples.append({
            "user_input": q,
            "response": answer,
            "reference": gt,
            "retrieved_contexts": contexts,
        })
        print(f"  [{i+1}/{total}] {elapsed:.1f}s | {q[:60]}...")

    return {
        "config": asdict(cfg),
        "samples": samples,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── RAGAS-оценка ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_with_ragas(samples: list[dict]) -> tuple[dict, pd.DataFrame]:
    """
    Оценивает samples через RAGAS.
    Возвращает (aggregate_scores, per_question_df).
    """
    (
        EvaluationDataset, evaluate,
        Faithfulness, ResponseRelevancy, LLMContextRecall, LLMContextPrecisionWithReference,
        LangchainLLMWrapper, ChatOpenAI,
    ) = _import_ragas()

    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import OpenAIEmbeddings

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    evaluator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    dataset = EvaluationDataset.from_list(samples)

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_emb),
        LLMContextRecall(llm=evaluator_llm),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
    ]

    print("[RAGAS] Запуск оценки...")
    result = evaluate(dataset=dataset, metrics=metrics)

    df = result.to_pandas()
    # Нормализуем имена столбцов
    rename_map = {"llm_context_precision_with_reference": "context_precision"}
    df = df.rename(columns=rename_map)
    scores = {
        "faithfulness": round(float(df["faithfulness"].mean()), 4),
        "answer_relevancy": round(float(df["answer_relevancy"].mean()), 4),
        "context_recall": round(float(df["context_recall"].mean()), 4),
        "context_precision": round(float(df["context_precision"].mean()), 4),
    }
    print(f"[RAGAS] Результаты: {scores}")
    return scores, df


# ══════════════════════════════════════════════════════════════════════════════
# ── Сводная таблица и сохранение ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def build_summary_table(results: list[dict]) -> pd.DataFrame:
    """Строит сводную таблицу из списка результатов экспериментов."""
    rows = []
    for r in results:
        cfg = r["config"]
        scores = r.get("scores", {})
        rows.append({
            "experiment": cfg["name"],
            "method": cfg["method"],
            "chunk_size": cfg["chunk_size"],
            "chunk_overlap": cfg["chunk_overlap"],
            "top_k": cfg["top_k"],
            "rrf_alpha": cfg["rrf_alpha"],
            "reranking": cfg["use_reranking"],
            "rerank_threshold": cfg.get("rerank_threshold", None),
            "faithfulness": scores.get("faithfulness", None),
            "answer_relevancy": scores.get("answer_relevancy", None),
            "context_recall": scores.get("context_recall", None),
            "context_precision": scores.get("context_precision", None),
        })
    return pd.DataFrame(rows)


def save_results(results: list[dict], path: str = "experiment_results.json") -> None:
    """Сохраняет результаты экспериментов в JSON."""
    # Убираем per_question_df (не сериализуется) — сохраняем только scores и samples
    serializable = []
    for r in results:
        entry = {
            "config": r["config"],
            "scores": r.get("scores", {}),
            "samples": r.get("samples", []),
        }
        serializable.append(entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"[Save] Результаты сохранены в {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ── Визуализации ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_recall",
               "context_precision"]
METRIC_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]


def plot_group_comparison(results: list[dict], group_name: str, out_dir: str = "plots") -> None:
    """Grouped bar chart: сравнение экспериментов внутри одной группы."""
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)

    labels = [r["config"]["name"] for r in results]
    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2.5), 5))
    for i, (m, color) in enumerate(zip(METRIC_COLS, METRIC_COLORS)):
        vals = [r["scores"].get(m, 0) for r in results]
        bars = ax.bar(x + i * width, vals, width, label=m, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Score")
    ax.set_title(f"Эксперимент: {group_name}")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"group_{group_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] {path}")


def plot_heatmap(summary_df: pd.DataFrame, out_dir: str = "plots") -> None:
    """Heatmap: все эксперименты × все метрики."""
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)

    experiments = summary_df["experiment"].tolist()
    data = summary_df[METRIC_COLS].values.astype(float)

    fig, ax = plt.subplots(figsize=(10, max(4, len(experiments) * 0.45)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(METRIC_COLS)))
    ax.set_xticklabels(METRIC_COLS, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels(experiments, fontsize=9)

    for i in range(len(experiments)):
        for j in range(len(METRIC_COLS)):
            val = data[i, j]
            if not np.isnan(val):
                color = "black" if 0.3 < val < 0.8 else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color=color, fontsize=8, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Score", shrink=0.8)
    ax.set_title("RAGAS Metrics Heatmap — все эксперименты", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "heatmap_all.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] {path}")


def plot_greedy_progression(best_history: list[dict], out_dir: str = "plots") -> None:
    """Line chart: прогрессия лучшего результата по группам greedy search."""
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    group_labels = [h["name"] for h in best_history]

    fig, ax = plt.subplots(figsize=(10, 5))
    for m, color in zip(METRIC_COLS, METRIC_COLORS):
        vals = [h["scores"].get(m, 0) for h in best_history]
        ax.plot(group_labels, vals, marker="o", linewidth=2, label=m, color=color)

    # Average
    avgs = [sum(h["scores"].get(m, 0) for m in METRIC_COLS) / len(METRIC_COLS)
            for h in best_history]
    ax.plot(group_labels, avgs, marker="s", linewidth=3, linestyle="--",
            color="black", label="Average", zorder=10)

    ax.set_ylabel("Score")
    ax.set_title("Прогрессия Greedy Search: лучший на каждом этапе")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "greedy_progression.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] {path}")


def plot_radar(best_result: dict, out_dir: str = "plots") -> None:
    """Radar chart: профиль лучшей конфигурации по 5 метрикам."""
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)

    scores = best_result["scores"]
    values = [scores.get(m, 0) for m in METRIC_COLS]
    values += values[:1]  # замыкаем полигон

    angles = np.linspace(0, 2 * np.pi, len(METRIC_COLS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color="#3498db", alpha=0.25)
    ax.plot(angles, values, color="#3498db", linewidth=2, marker="o")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_COLS, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(f"Лучшая конфигурация: {best_result['config']['name']}", fontsize=12, pad=20)

    for angle, val, label in zip(angles[:-1], values[:-1], METRIC_COLS):
        ax.text(angle, val + 0.05, f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "radar_best.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] {path}")


def plot_per_question_difficulty(results: list[dict], out_dir: str = "plots") -> None:
    """Horizontal bar: средний faithfulness по каждому вопросу (самые сложные вопросы)."""
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)

    question_scores: dict[str, list[float]] = {}
    for r in results:
        if "per_question_df" not in r:
            continue
        df = r["per_question_df"]
        if "faithfulness" not in df.columns:
            continue
        for _, row in df.iterrows():
            q = row.get("user_input", "")[:60]
            val = row.get("faithfulness", float("nan"))
            if q and not pd.isna(val):
                question_scores.setdefault(q, []).append(val)

    if not question_scores:
        return

    avg_scores = {q: sum(v) / len(v) for q, v in question_scores.items()}
    sorted_qs = sorted(avg_scores, key=avg_scores.get)

    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_qs) * 0.35)))
    colors = ["#e74c3c" if avg_scores[q] < 0.5 else "#f39c12" if avg_scores[q] < 0.8 else "#2ecc71"
              for q in sorted_qs]
    y = range(len(sorted_qs))
    ax.barh(y, [avg_scores[q] for q in sorted_qs], color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([q + "..." for q in sorted_qs], fontsize=7)
    ax.set_xlabel("Avg Faithfulness")
    ax.set_title("Сложность вопросов (красный = низкий faithfulness)")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "question_difficulty.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] {path}")


def generate_all_plots(results: list[dict], summary_df: pd.DataFrame,
                       best_history: list[dict],
                       grouped_results: list[list[dict]] | None = None) -> None:
    """Генерирует heatmap из результатов экспериментов."""
    import matplotlib
    matplotlib.use("Agg")  # без GUI
    plot_heatmap(summary_df)


# ══════════════════════════════════════════════════════════════════════════════
# ── Матрица экспериментов (greedy search) ────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def build_experiment_matrix() -> list[list[ExperimentConfig]]:
    """
    Возвращает 6 групп экспериментов для greedy search.
    Winner каждого этапа наследуется следующим.
    """
    groups = []

    # Группа 0: Baseline
    groups.append([
        ExperimentConfig(name="baseline", method="recursive",
                         chunk_size=1024, chunk_overlap=200,
                         top_k=5, rrf_alpha=0.5, use_reranking=True),
    ])

    # Группа 1: Chunk Size
    groups.append([
        ExperimentConfig(name="chunk_512", method="recursive",
                         chunk_size=512, chunk_overlap=100,
                         top_k=5, rrf_alpha=0.5, use_reranking=True,
                         collection_name="rag_recursive_512"),
        ExperimentConfig(name="chunk_1024", method="recursive",
                         chunk_size=1024, chunk_overlap=200,
                         top_k=5, rrf_alpha=0.5, use_reranking=True),
        ExperimentConfig(name="chunk_2048", method="recursive",
                         chunk_size=2048, chunk_overlap=200,
                         top_k=5, rrf_alpha=0.5, use_reranking=True,
                         collection_name="rag_recursive_2048"),
    ])

    # Группа 2: Top-K (наследует best chunk_size)
    groups.append([
        ExperimentConfig(name="topk_3", top_k=3),
        ExperimentConfig(name="topk_5", top_k=5),
        ExperimentConfig(name="topk_10", top_k=10),
    ])

    # Группа 3: Alpha (RRF) (наследует best chunk_size + top_k)
    groups.append([
        ExperimentConfig(name="alpha_0.0", rrf_alpha=0.0),
        ExperimentConfig(name="alpha_0.3", rrf_alpha=0.3),
        ExperimentConfig(name="alpha_0.5", rrf_alpha=0.5),
        ExperimentConfig(name="alpha_0.7", rrf_alpha=0.7),
        ExperimentConfig(name="alpha_1.0", rrf_alpha=1.0),
    ])

    # Группа 4: Reranking (наследует все предыдущие best)
    groups.append([
        ExperimentConfig(name="rerank_on", use_reranking=True),
        ExperimentConfig(name="rerank_off", use_reranking=False),
    ])

    # Группа 5: Chunking strategy (наследует best гиперпараметры)
    groups.append([
        ExperimentConfig(name="strategy_fixed", method="fixed"),
        ExperimentConfig(name="strategy_recursive", method="recursive"),
        ExperimentConfig(name="strategy_layout", method="layout"),
        ExperimentConfig(name="strategy_semantic", method="semantic"),
    ])

    # Группа 6: Rerank Threshold (наследует все предыдущие best)
    groups.append([
        ExperimentConfig(name="rerank_thresh_0.0",  rerank_threshold=0.0),
        ExperimentConfig(name="rerank_thresh_0.01", rerank_threshold=0.01),
        ExperimentConfig(name="rerank_thresh_0.1",  rerank_threshold=0.1),
        ExperimentConfig(name="rerank_thresh_0.3",  rerank_threshold=0.3),
    ])

    return groups


def _detect_varied_keys(configs: list[ExperimentConfig]) -> set[str]:
    """Определяет какие параметры варьируются в группе (имеют разные значения)."""
    if len(configs) <= 1:
        return set()
    fields = ["method", "chunk_size", "chunk_overlap", "top_k",
              "rrf_alpha", "use_reranking", "collection_name", "rerank_threshold"]
    varied = set()
    for key in fields:
        values = {getattr(cfg, key) for cfg in configs}
        if len(values) > 1:
            varied.add(key)
    return varied


def apply_best_params(configs: list[ExperimentConfig], best: dict) -> list[ExperimentConfig]:
    """Применяет лучшие параметры предыдущего этапа к конфигурациям текущей группы.

    Параметры, которые варьируются в текущей группе, НЕ перезаписываются,
    иначе greedy search теряет смысл — все конфигурации станут одинаковыми.
    """
    varied = _detect_varied_keys(configs)
    for cfg in configs:
        for key, val in best.items():
            if not hasattr(cfg, key):
                continue
            if key in varied:
                continue  # не трогаем варьируемый параметр
            setattr(cfg, key, val)
    return configs


def run_greedy_search(
    golden: list[dict],
    groups: list[list[ExperimentConfig]],
    skip_reindex: bool = False,
) -> tuple[list[dict], list[dict], list[list[dict]]]:
    """
    Запускает greedy search по группам экспериментов.
    Каждая группа оценивается, лучший результат наследуется следующей группе.
    Возвращает (all_results, best_score_history, grouped_results).
    """
    all_results = []
    best_params: dict = {}
    best_score_history: list[dict] = []
    grouped_results: list[list[dict]] = []

    for group_idx, group in enumerate(groups):
        print(f"\n{'#'*60}")
        print(f"  ГРУППА {group_idx}: {[c.name for c in group]}")
        print(f"  Best params so far: {best_params}")
        print(f"{'#'*60}")

        # Применяем лучшие параметры предыдущих этапов
        if group_idx >= 2 and best_params:
            group = apply_best_params(group, best_params)

        group_results = []
        for cfg in group:
            # Переиндексация если нужна кастомная коллекция
            if cfg.collection_name and not skip_reindex:
                _reindex_for_experiment(cfg)

            result = run_experiment(cfg, golden)
            scores, per_q_df = evaluate_with_ragas(result["samples"])
            result["scores"] = scores
            result["per_question_df"] = per_q_df
            group_results.append(result)
            all_results.append(result)

        grouped_results.append(group_results)

        # Определяем лучший эксперимент в группе (по среднему 5 метрик)
        best_result = max(
            group_results,
            key=lambda r: sum(r["scores"].values()) / len(r["scores"]),
        )
        best_cfg = best_result["config"]
        avg = sum(best_result["scores"].values()) / len(best_result["scores"])

        print(f"\n  >>> Лучший в группе {group_idx}: {best_cfg['name']} "
              f"(avg={avg:.4f}, scores={best_result['scores']})")

        # Наследуем лучшие параметры
        for key in ["method", "chunk_size", "chunk_overlap", "top_k", "rrf_alpha",
                     "use_reranking", "collection_name", "rerank_threshold"]:
            best_params[key] = best_cfg[key]

        best_score_history.append({
            "group": group_idx,
            "name": best_cfg["name"],
            "scores": best_result["scores"],
        })

    # Печатаем прогрессию
    print(f"\n{'='*60}")
    print("  ПРОГРЕССИЯ GREEDY SEARCH")
    print(f"{'='*60}")
    for h in best_score_history:
        avg = sum(h["scores"].values()) / len(h["scores"])
        print(f"  Группа {h['group']}: {h['name']} → avg={avg:.4f} {h['scores']}")

    return all_results, best_score_history, grouped_results


def _reindex_for_experiment(cfg: ExperimentConfig) -> None:
    """Переиндексирует коллекцию для эксперимента с нестандартным chunk_size."""
    from index_llama import run_pipeline
    print(f"[Reindex] Создание коллекции '{cfg.collection_name}' "
          f"(chunk_size={cfg.chunk_size}, overlap={cfg.chunk_overlap})...")
    run_pipeline(
        method=cfg.method,
        reset=True,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        collection_name=cfg.collection_name,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── CLI ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Запуск экспериментов и RAGAS-оценки")
    parser.add_argument(
        "--experiments", nargs="*", type=int,
        help="Индексы групп экспериментов (0-5). Без аргумента — все.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Быстрый тест (первые 3 вопроса из golden dataset).",
    )
    parser.add_argument(
        "--skip-reindex", action="store_true",
        help="Пропустить переиндексацию (если коллекции уже созданы).",
    )
    args = parser.parse_args()

    golden = load_golden_dataset()
    if args.quick:
        golden = golden[:3]
        print(f"[Quick] Используем только {len(golden)} вопросов.")

    groups = build_experiment_matrix()
    if args.experiments is not None:
        groups = [groups[i] for i in args.experiments if i < len(groups)]

    results, best_history, grouped_results = run_greedy_search(golden, groups, skip_reindex=args.skip_reindex)

    # Сводная таблица
    summary = build_summary_table(results)
    print(f"\n{'='*60}")
    print("  СВОДНАЯ ТАБЛИЦА")
    print(f"{'='*60}")
    print(summary.to_string(index=False))

    # Сохранение
    save_results(results)

    # Визуализации
    generate_all_plots(results, summary, best_history, grouped_results)

    # Динамические итоги
    print_conclusions(results, best_history, grouped_results)


def print_conclusions(results: list[dict], best_history: list[dict],
                      grouped_results: list[list[dict]]) -> None:
    """Автоматически формирует текстовые выводы из результатов экспериментов."""
    if not results:
        return

    print(f"\n{'='*60}")
    print("  ИТОГОВЫЕ ВЫВОДЫ (автоматически)")
    print(f"{'='*60}")

    # 1. Лучшая конфигурация
    best = max(results, key=lambda r: sum(r.get("scores", {}).values()))
    best_cfg = best["config"]
    best_avg = sum(best["scores"].values()) / len(best["scores"])
    print(f"\n1. ЛУЧШАЯ КОНФИГУРАЦИЯ: {best_cfg['name']} (avg={best_avg:.4f})")
    print(f"   method={best_cfg['method']}, chunk={best_cfg['chunk_size']}/{best_cfg['chunk_overlap']}, "
          f"top_k={best_cfg['top_k']}, alpha={best_cfg['rrf_alpha']}, rerank={best_cfg['use_reranking']}")
    for m, v in best["scores"].items():
        print(f"   {m}: {v:.4f}")

    # 2. Самый влиятельный параметр (группа с наибольшим разбросом avg score)
    if len(grouped_results) > 1:
        max_spread = 0
        most_influential = ""
        least_spread = float("inf")
        least_influential = ""
        for i, group in enumerate(grouped_results):
            if len(group) < 2:
                continue
            avgs = [sum(r["scores"].values()) / len(r["scores"]) for r in group]
            spread = max(avgs) - min(avgs)
            name = group[0]["config"]["name"].split("_")[0]
            if spread > max_spread:
                max_spread = spread
                most_influential = name
            if spread < least_spread:
                least_spread = spread
                least_influential = name

        print(f"\n2. САМЫЙ ВЛИЯТЕЛЬНЫЙ ПАРАМЕТР: {most_influential} (разброс avg: {max_spread:.4f})")
        print(f"3. НАИМЕНЕЕ ВЛИЯТЕЛЬНЫЙ ПАРАМЕТР: {least_influential} (разброс avg: {least_spread:.4f})")

    # 4. Alpha анализ
    alpha_results = [r for r in results if r["config"]["name"].startswith("alpha_")]
    if alpha_results:
        best_alpha = max(alpha_results, key=lambda r: sum(r["scores"].values()))
        print(f"\n4. ЛУЧШИЙ ALPHA: {best_alpha['config']['rrf_alpha']} "
              f"(avg={sum(best_alpha['scores'].values()) / len(best_alpha['scores']):.4f})")
        if best_alpha["config"]["rrf_alpha"] == 0.0:
            print("   → Только BM25 — ключевые слова важнее семантики для этих документов")
        elif best_alpha["config"]["rrf_alpha"] == 1.0:
            print("   → Только Vector — семантический поиск доминирует")
        else:
            print(f"   → Гибрид vector/BM25 в пропорции {best_alpha['config']['rrf_alpha']:.1f}/{1-best_alpha['config']['rrf_alpha']:.1f}")

    # 5. Reranking
    rerank_on = [r for r in results if r["config"]["name"] == "rerank_on"]
    rerank_off = [r for r in results if r["config"]["name"] == "rerank_off"]
    if rerank_on and rerank_off:
        avg_on = sum(rerank_on[0]["scores"].values()) / len(rerank_on[0]["scores"])
        avg_off = sum(rerank_off[0]["scores"].values()) / len(rerank_off[0]["scores"])
        diff = avg_on - avg_off
        if diff > 0:
            print(f"\n5. RERANKING: помог (+{diff:.4f} avg)")
        else:
            print(f"\n5. RERANKING: не помог ({diff:.4f} avg)")
        print(f"   С реранкингом:  avg={avg_on:.4f}")
        print(f"   Без реранкинга: avg={avg_off:.4f}")

    # 6. Rerank Threshold
    thresh_results = [r for r in results if r["config"]["name"].startswith("rerank_thresh_")]
    if thresh_results:
        best_thresh = max(thresh_results, key=lambda r: sum(r["scores"].values()))
        avg_best = sum(best_thresh["scores"].values()) / len(best_thresh["scores"])
        t_val = best_thresh["config"].get("rerank_threshold", "?")
        print(f"\n6. ЛУЧШИЙ RERANK THRESHOLD: {t_val} (avg={avg_best:.4f})")
        if t_val == 0.0:
            print("   → Без фильтрации — CrossEncoder ранжирует, порог не нужен")
        elif t_val <= 0.01:
            print("   → Минимальный порог — почти без фильтрации")
        else:
            print(f"   → Порог {t_val} отсекает низкоскоринговые чанки")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
