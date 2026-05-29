"""
Generate agent_memory_qasper_ctx8192_episode_safe_seed42.jsonl from allenai/qasper.

Each paper becomes one multi-turn conversation:
  - User turn 0: "Read and remember this paper: <full text>"
  - Assistant turn 0: "Understood."
  - User turn N: question N
  - Assistant turn N: answer N

"episode_safe" means:
  - >= 2 answerable QA pairs (need history to write + at least one prediction)
  - paper full text fits within write budget (ctx_tokens <= max_write_tokens)

We take the shortest 2,219 conversations by total token count (seed=42 for tie-breaking).

Usage:
    python scripts/generate_qasper_train_data.py \
        --output /path/to/agent_memory_qasper_ctx8192_episode_safe_seed42.jsonl \
        [--max-write-tokens 8192] [--min-qa-pairs 2] [--n-samples 2219] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path


def _rough_token_count(text: str) -> int:
    """Whitespace-split token count; ~10% over BPE but fast and bias-consistent."""
    return len(text.split())


def _extract_answer(answer_entry: dict) -> str | None:
    """Return the best string answer for one annotator's answer entry, or None if unanswerable."""
    if answer_entry.get("unanswerable"):
        return None
    free = (answer_entry.get("free_form_answer") or "").strip()
    if free:
        return free
    spans = answer_entry.get("extractive_spans") or []
    if spans:
        return " ... ".join(s.strip() for s in spans if s.strip())
    yn = answer_entry.get("yes_no")
    if yn is not None:
        return "Yes" if yn else "No"
    return None


def _build_paper_text(example: dict) -> str:
    """Concatenate title, abstract, and section paragraphs into a single string.

    Handles both raw QASPER JSON (full_text = list of section dicts)
    and HuggingFace dataset format (full_text = dict with section_name/paragraphs lists).
    """
    parts: list[str] = []
    title = (example.get("title") or "").strip()
    if title:
        parts.append(f"Title: {title}")
    abstract = (example.get("abstract") or "").strip()
    if abstract:
        parts.append(f"Abstract: {abstract}")
    full_text = example.get("full_text") or []
    # Raw JSON: list of {"section_name": str, "paragraphs": list[str]}
    if isinstance(full_text, list):
        sections = full_text
    else:
        # HuggingFace format: {"section_name": [...], "paragraphs": [[...]]}
        section_names = full_text.get("section_name") or []
        paragraphs_list = full_text.get("paragraphs") or []
        sections = [
            {"section_name": s, "paragraphs": p}
            for s, p in zip(section_names, paragraphs_list)
        ]
    for sec in sections:
        sec_name = (sec.get("section_name") or "").strip()
        if sec_name:
            parts.append(f"\n## {sec_name}")
        for para in (sec.get("paragraphs") or []):
            para = (para or "").strip()
            if para:
                parts.append(para)
    return "\n\n".join(parts)


def _build_conversation(paper_text: str, qa_pairs: list[tuple[str, str]]) -> list[dict]:
    """Build messages list from paper text and QA pairs."""
    messages = [
        {
            "role": "user",
            "content": (
                "Please read and remember the following research paper carefully. "
                "You will be asked questions about it.\n\n" + paper_text
            ),
        },
        {"role": "assistant", "content": "Understood. I have read and memorized the paper."},
    ]
    for question, answer in qa_pairs:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    return messages


def process_dataset(
    examples,
    max_write_tokens: int,
    min_qa_pairs: int,
) -> list[dict]:
    """Convert QASPER examples to conversation dicts with length metadata."""
    results = []
    for ex in examples:
        paper_text = _build_paper_text(ex)
        paper_tokens = _rough_token_count(paper_text)
        if paper_tokens > max_write_tokens:
            continue

        qas_raw = ex.get("qas") or []
        # Raw JSON: list of QA dicts {"question": str, "answers": [...], ...}
        # HF format: dict with "question": [...], "answers": [...] parallel lists
        if isinstance(qas_raw, list):
            questions = [qa.get("question", "") for qa in qas_raw]
            answers_list = [qa.get("answers", []) for qa in qas_raw]
        else:
            questions = qas_raw.get("question") or []
            answers_list = qas_raw.get("answers") or []

        qa_pairs: list[tuple[str, str]] = []
        for q, ans_group in zip(questions, answers_list):
            q = (q or "").strip()
            if not q:
                continue
            # Handle both raw JSON and HuggingFace formats.
            # Raw JSON:  ans_group = [{"answer": dict, "annotation_id": ..., "worker_id": ...}, ...]
            # HF format: ans_group = {"answer": [list of dicts], "annotation_id": ..., "worker_id": ...}
            best_answer = None
            if isinstance(ans_group, list):
                # Raw JSON: each element is one annotator's response
                for annotator in ans_group:
                    answer_dict = annotator.get("answer") if isinstance(annotator, dict) else None
                    if answer_dict:
                        best_answer = _extract_answer(answer_dict)
                        if best_answer:
                            break
            else:
                # HF format: dict with "answer" = list of answer dicts
                annotator_answers = (ans_group or {}).get("answer") or []
                for ann in annotator_answers:
                    best_answer = _extract_answer(ann)
                    if best_answer:
                        break
            if best_answer is None:
                continue
            qa_pairs.append((q, best_answer))

        if len(qa_pairs) < min_qa_pairs:
            continue

        messages = _build_conversation(paper_text, qa_pairs)
        total_tokens = sum(_rough_token_count(m["content"]) for m in messages)
        results.append({"messages": messages, "_total_tokens": total_tokens})

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QASPER episode training data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("agent_memory_qasper_ctx8192_episode_safe_seed42.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--max-write-tokens", type=int, default=8192)
    parser.add_argument("--min-qa-pairs", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="HuggingFace datasets cache directory",
    )
    parser.add_argument(
        "--qasper-json",
        type=Path,
        default=None,
        help="Path to raw QASPER JSON file (qasper-train-v0.3.json). "
             "Use this instead of --hf-cache-dir when HuggingFace is unreachable.",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    # Support reading from a local raw QASPER JSON file (dict keyed by paper_id)
    if args.qasper_json is not None:
        print(f"Loading QASPER from local file: {args.qasper_json}")
        with open(args.qasper_json, encoding="utf-8") as f:
            raw = json.load(f)
        # Convert dict-keyed format to list of dicts
        examples = [{"id": pid, **paper} for pid, paper in raw.items()]
        print(f"  Loaded {len(examples)} papers")
    else:
        print("Loading allenai/qasper train split...")
        cache_dir = str(args.hf_cache_dir) if args.hf_cache_dir else None
        ds = load_dataset("allenai/qasper", split="train", cache_dir=cache_dir)
        examples = list(ds)
        print(f"  Loaded {len(examples)} papers")

    print("Converting to episode-safe conversations...")
    candidates = process_dataset(examples, args.max_write_tokens, args.min_qa_pairs)
    print(f"  Episode-safe candidates: {len(candidates)}")

    if len(candidates) < args.n_samples:
        print(
            f"WARNING: only {len(candidates)} candidates, fewer than requested {args.n_samples}",
            file=sys.stderr,
        )

    # Sort by total token count (ascending), break ties deterministically with seed
    rng = random.Random(args.seed)
    candidates.sort(key=lambda x: (x["_total_tokens"], rng.random()))

    selected = candidates[: args.n_samples]
    print(
        f"  Selected {len(selected)} shortest conversations "
        f"(token range: {selected[0]['_total_tokens']}–{selected[-1]['_total_tokens']})"
    )

    # Strip internal metadata key before writing
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in selected:
            out = {"messages": item["messages"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} examples to {args.output}")


if __name__ == "__main__":
    main()
