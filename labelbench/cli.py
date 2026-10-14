from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adjudication import (
    aggregate_submissions,
    build_offline_review_bundle,
    create_multilabel_review_tasks,
)
from .audit import audit_normalized_splits
from .datasets import (
    ARATOX_LABELS,
    deterministic_sample,
    fetch_aratox_binary,
    fetch_aratox_multilabel,
    fetch_iadd_dart_region_eval,
    fetch_iadd_dart_region_train,
    fetch_massive_ar_sa,
    read_multilabel_examples,
    write_multilabel_examples,
    write_examples,
)
from .comparisons import compare_prediction_files
from .models import CharacterNgramNB, MajorityClassifier, OneVsRestCharacterNgram
from .multilabel_runner import run_multilabel_classifier
from .ollama import (
    OllamaDialectRegionClassifier,
    OllamaMultiLabelToxicityClassifier,
    OllamaToxicityClassifier,
)
from .openai_api import OpenAIResponsesDialectClassifier
from .paths import CACHE_ENV_VAR, default_cache_dir
from .runner import run_classifier
from .reporting import build_release


def _add_cache_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "directory for normalized manifests; raw downloads go to its "
            f"sibling raw/ directory (default: ${CACHE_ENV_VAR}/normalized, "
            "or labelbench/normalized in the user cache directory)"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="labelbench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("pilot", help="run the MASSIVE ar-SA vertical slice")
    pilot.add_argument("--train-limit", type=int, default=5000)
    pilot.add_argument("--test-limit", type=int, default=1000)
    pilot.add_argument("--output", type=Path, required=True)
    _add_cache_dir_argument(pilot)
    pilot.add_argument("--seed", type=int, default=42)
    aratox = subparsers.add_parser(
        "pilot-aratox", help="run the native AraTox binary vertical slice"
    )
    aratox.add_argument("--train-limit", type=int)
    aratox.add_argument("--test-limit", type=int)
    aratox.add_argument(
        "--random-test",
        action="store_true",
        help="sample the held-out split with --seed instead of taking a prefix",
    )
    aratox.add_argument("--output", type=Path, required=True)
    _add_cache_dir_argument(aratox)
    aratox.add_argument("--seed", type=int, default=42)
    audit = subparsers.add_parser(
        "audit", help="audit normalized train/test files for leakage and conflicts"
    )
    audit.add_argument("--train", type=Path, required=True)
    audit.add_argument("--test", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    ollama = subparsers.add_parser(
        "pilot-ollama-aratox",
        help="run a local Ollama model on a held-out AraTox slice",
    )
    ollama.add_argument("--model", default="qwen2.5:3b")
    ollama.add_argument("--test-limit", type=int, default=200)
    ollama.add_argument("--output", type=Path, required=True)
    _add_cache_dir_argument(ollama)
    ollama.add_argument("--seed", type=int, default=42)
    ollama.add_argument(
        "--confidence-mode",
        choices=("constant", "verbalized"),
        default="constant",
        help="constant preserves v0.1 behavior; verbalized asks the model for a 0-1 confidence score",
    )
    aratox_multilabel = subparsers.add_parser(
        "pilot-aratox-multilabel",
        help="run the full AraTox seven-label local baseline",
    )
    aratox_multilabel.add_argument("--train-limit", type=int)
    aratox_multilabel.add_argument("--test-limit", type=int)
    aratox_multilabel.add_argument(
        "--random-test",
        action="store_true",
        help="sample the held-out split with --seed instead of taking a prefix",
    )
    aratox_multilabel.add_argument("--output", type=Path, required=True)
    _add_cache_dir_argument(aratox_multilabel)
    aratox_multilabel.add_argument("--seed", type=int, default=42)
    aratox_multilabel.add_argument(
        "--tune-thresholds",
        action="store_true",
        help="tune per-label thresholds on a held-out training partition",
    )
    ollama_multilabel = subparsers.add_parser(
        "pilot-ollama-aratox-multilabel",
        help="run a local Ollama model on a sampled full AraTox task",
    )
    ollama_multilabel.add_argument("--model", default="qwen2.5:3b")
    ollama_multilabel.add_argument("--test-limit", type=int, default=300)
    ollama_multilabel.add_argument("--output", type=Path, required=True)
    _add_cache_dir_argument(ollama_multilabel)
    ollama_multilabel.add_argument("--seed", type=int, default=42)
    ollama_multilabel.add_argument(
        "--confidence-mode",
        choices=("constant", "verbalized"),
        default="constant",
        help="constant preserves v0.1 behavior; verbalized asks for exact-set confidence",
    )
    ollama_iadd = subparsers.add_parser(
        "pilot-ollama-iadd",
        help="run a local Ollama model on a balanced IADD DART region manifest",
    )
    ollama_iadd.add_argument("--model", default="qwen2.5:3b")
    ollama_iadd.add_argument("--per-region", type=int, default=100)
    ollama_iadd.add_argument("--output", type=Path, required=True)
    _add_cache_dir_argument(ollama_iadd)
    ollama_iadd.add_argument("--seed", type=int, default=42)
    ollama_iadd.add_argument(
        "--confidence-mode",
        choices=("constant", "verbalized"),
        default="constant",
        help="constant preserves v0.1 behavior; verbalized asks the model for a 0-1 confidence score",
    )
    iadd_supervised = subparsers.add_parser(
        "pilot-iadd-supervised",
        help="run a disjoint balanced DART supervised diagnostic",
    )
    iadd_supervised.add_argument("--train-per-region", type=int, default=100)
    iadd_supervised.add_argument("--test-per-region", type=int, default=100)
    iadd_supervised.add_argument("--output", type=Path, required=True)
    _add_cache_dir_argument(iadd_supervised)
    iadd_supervised.add_argument("--seed", type=int, default=42)
    openai_iadd = subparsers.add_parser(
        "pilot-openai-iadd",
        help="run an OpenAI Responses model on the fixed IADD region manifest",
    )
    openai_iadd.add_argument(
        "--model",
        default="gpt-5.4-mini-2026-03-17",
        help="use a dated snapshot for reproducible public results",
    )
    openai_iadd.add_argument("--per-region", type=int, default=100)
    openai_iadd.add_argument("--output", type=Path, required=True)
    _add_cache_dir_argument(openai_iadd)
    openai_iadd.add_argument("--seed", type=int, default=42)
    openai_iadd.add_argument(
        "--confidence-mode",
        choices=("constant", "verbalized"),
        default="verbalized",
        help="verbalized asks for a structured 0-1 confidence score",
    )
    review = subparsers.add_parser(
        "make-review-bundle",
        help="create a blinded offline review bundle from multi-label examples",
    )
    review.add_argument("--examples", type=Path, required=True)
    review.add_argument("--manifest", type=Path, required=True)
    review.add_argument("--bundle", type=Path, required=True)
    review.add_argument(
        "--template",
        type=Path,
        default=None,
        help="custom review template (default: the template shipped with labelbench)",
    )
    review.add_argument("--seed", type=int, default=42)
    aggregate = subparsers.add_parser(
        "aggregate-reviews",
        help="validate and aggregate blinded reviewer submissions",
    )
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--submissions", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    release = subparsers.add_parser(
        "build-release",
        help="build publishable tables and charts from immutable run reports",
    )
    release.add_argument("--reports", type=Path, nargs="+", required=True)
    release.add_argument("--output", type=Path, required=True)
    release.add_argument("--release-id", required=True)
    release.add_argument("--comparisons", type=Path, nargs="*", default=[])
    comparison = subparsers.add_parser(
        "compare-runs",
        help="run paired bootstrap and exact McNemar tests on two prediction logs",
    )
    comparison.add_argument("--a", type=Path, required=True)
    comparison.add_argument("--b", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--iterations", type=int, default=10_000)
    comparison.add_argument("--seed", type=int, default=42)
    return parser


def run_baselines(
    *,
    train,
    test,
    train_file,
    test_file,
    output_dir: Path,
    dataset_id: str,
    task: str,
    seed: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for classifier in (MajorityClassifier(), CharacterNgramNB()):
        report = run_classifier(
            classifier=classifier,
            train=train,
            test=test,
            output_dir=output_dir,
            train_file=train_file,
            test_file=test_file,
            dataset_id=dataset_id,
            task=task,
            seed=seed,
        )
        reports.append(
            {
                "model": report["model"],
                "accuracy": report["metrics"]["accuracy"],
                "macro_f1": report["metrics"]["macro_f1"],
                "sac_95": report["metrics"]["sac_95"]["coverage"],
                "sac_98": report["metrics"]["sac_98"]["coverage"],
            }
        )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"results": reports}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"results": reports}, ensure_ascii=False, indent=2))
    return 0


def run_massive_pilot(args: argparse.Namespace) -> int:
    output_dir: Path = args.output
    train, train_file = fetch_massive_ar_sa(
        "train", args.train_limit, args.cache_dir
    )
    test, test_file = fetch_massive_ar_sa("test", args.test_limit, args.cache_dir)
    return run_baselines(
        train=train,
        test=test,
        train_file=train_file,
        test_file=test_file,
        output_dir=output_dir,
        dataset_id="massive_ar_sa",
        task="intent_classification",
        seed=args.seed,
    )


def run_aratox_pilot(args: argparse.Namespace) -> int:
    train, train_file = fetch_aratox_binary(
        "train", args.train_limit, args.cache_dir
    )
    if args.random_test:
        if args.test_limit is None:
            raise ValueError("--random-test requires --test-limit")
        all_test, _ = fetch_aratox_binary("test", None, args.cache_dir)
        test = deterministic_sample(all_test, args.test_limit, args.seed)
        test_file = (
            args.cache_dir
            / f"aratox_binary_test_sample{args.test_limit}_seed{args.seed}.jsonl"
        )
        if not test_file.exists():
            write_examples(test_file, test)
    else:
        test, test_file = fetch_aratox_binary(
            "test", args.test_limit, args.cache_dir
        )
    return run_baselines(
        train=train,
        test=test,
        train_file=train_file,
        test_file=test_file,
        output_dir=args.output,
        dataset_id="aratox",
        task="binary_toxicity",
        seed=args.seed,
    )


def run_ollama_aratox_pilot(args: argparse.Namespace) -> int:
    train, train_file = fetch_aratox_binary("train", 1, args.cache_dir)
    all_test, _ = fetch_aratox_binary("test", None, args.cache_dir)
    test = deterministic_sample(all_test, args.test_limit, args.seed)
    test_file = (
        args.cache_dir
        / f"aratox_binary_test_sample{args.test_limit}_seed{args.seed}.jsonl"
    )
    if not test_file.exists():
        write_examples(test_file, test)
    classifier = OllamaToxicityClassifier(
        model=args.model,
        seed=args.seed,
        confidence_mode=args.confidence_mode,
    )
    report = run_classifier(
        classifier=classifier,
        train=train,
        test=test,
        output_dir=args.output,
        train_file=train_file,
        test_file=test_file,
        dataset_id="aratox",
        task="binary_toxicity",
        seed=args.seed,
    )
    summary = {
        "model": report["model"],
        "accuracy": report["metrics"]["accuracy"],
        "macro_f1": report["metrics"]["macro_f1"],
        "sac_95": report["metrics"]["sac_95"]["coverage"],
        "sac_98": report["metrics"]["sac_98"]["coverage"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_aratox_multilabel_pilot(args: argparse.Namespace) -> int:
    train, train_file = fetch_aratox_multilabel(
        "train", args.train_limit, args.cache_dir
    )
    if args.random_test:
        if args.test_limit is None:
            raise ValueError("--random-test requires --test-limit")
        all_test, _ = fetch_aratox_multilabel("test", None, args.cache_dir)
        test = deterministic_sample(all_test, args.test_limit, args.seed)
        test_file = (
            args.cache_dir
            / f"aratox_multilabel_test_sample{args.test_limit}_seed{args.seed}.jsonl"
        )
        if not test_file.exists():
            write_multilabel_examples(test_file, test)
    else:
        test, test_file = fetch_aratox_multilabel(
            "test", args.test_limit, args.cache_dir
        )
    report = run_multilabel_classifier(
        classifier=OneVsRestCharacterNgram(
            tune_thresholds=args.tune_thresholds,
            seed=args.seed,
        ),
        train=train,
        test=test,
        labels=ARATOX_LABELS,
        output_dir=args.output,
        train_file=train_file,
        test_file=test_file,
        dataset_id="aratox",
        task="toxicity_multilabel",
        seed=args.seed,
    )
    summary = {
        "model": report["model"],
        "subset_accuracy": report["metrics"]["subset_accuracy"],
        "micro_f1": report["metrics"]["micro_f1"],
        "macro_f1": report["metrics"]["macro_f1"],
        "sac_95_exact_set": report["metrics"]["sac_95_exact_set"]["coverage"],
        "sac_98_exact_set": report["metrics"]["sac_98_exact_set"]["coverage"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_ollama_aratox_multilabel_pilot(args: argparse.Namespace) -> int:
    train, train_file = fetch_aratox_multilabel("train", 1, args.cache_dir)
    all_test, _ = fetch_aratox_multilabel("test", None, args.cache_dir)
    test = deterministic_sample(all_test, args.test_limit, args.seed)
    test_file = (
        args.cache_dir
        / f"aratox_multilabel_test_sample{args.test_limit}_seed{args.seed}.jsonl"
    )
    if not test_file.exists():
        write_multilabel_examples(test_file, test)
    report = run_multilabel_classifier(
        classifier=OllamaMultiLabelToxicityClassifier(
            model=args.model,
            seed=args.seed,
            confidence_mode=args.confidence_mode,
        ),
        train=train,
        test=test,
        labels=ARATOX_LABELS,
        output_dir=args.output,
        train_file=train_file,
        test_file=test_file,
        dataset_id="aratox",
        task="toxicity_multilabel",
        seed=args.seed,
    )
    summary = {
        "model": report["model"],
        "subset_accuracy": report["metrics"]["subset_accuracy"],
        "micro_f1": report["metrics"]["micro_f1"],
        "macro_f1": report["metrics"]["macro_f1"],
        "sac_95_exact_set": report["metrics"]["sac_95_exact_set"]["coverage"],
        "sac_98_exact_set": report["metrics"]["sac_98_exact_set"]["coverage"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_ollama_iadd_pilot(args: argparse.Namespace) -> int:
    test, test_file = fetch_iadd_dart_region_eval(
        cache_dir=args.cache_dir,
        per_region=args.per_region,
        seed=args.seed,
    )
    classifier = OllamaDialectRegionClassifier(
        model=args.model,
        seed=args.seed,
        confidence_mode=args.confidence_mode,
    )
    report = run_classifier(
        classifier=classifier,
        train=[],
        test=test,
        output_dir=args.output,
        train_file=None,
        test_file=test_file,
        dataset_id="iadd_dart_region",
        task="dialect_region_identification",
        seed=args.seed,
    )
    summary = {
        "model": report["model"],
        "examples": len(test),
        "accuracy": report["metrics"]["accuracy"],
        "macro_f1": report["metrics"]["macro_f1"],
        "sac_95": report["metrics"]["sac_95"]["coverage"],
        "sac_98": report["metrics"]["sac_98"]["coverage"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_iadd_supervised_pilot(args: argparse.Namespace) -> int:
    test, test_file = fetch_iadd_dart_region_eval(
        cache_dir=args.cache_dir,
        per_region=args.test_per_region,
        seed=args.seed,
    )
    train, train_file = fetch_iadd_dart_region_train(
        cache_dir=args.cache_dir,
        per_region=args.train_per_region,
        eval_per_region=args.test_per_region,
        seed=args.seed,
    )
    return run_baselines(
        train=train,
        test=test,
        train_file=train_file,
        test_file=test_file,
        output_dir=args.output,
        dataset_id="iadd_dart_region",
        task="dialect_region_identification",
        seed=args.seed,
    )


def run_openai_iadd_pilot(args: argparse.Namespace) -> int:
    test, test_file = fetch_iadd_dart_region_eval(
        cache_dir=args.cache_dir,
        per_region=args.per_region,
        seed=args.seed,
    )
    report = run_classifier(
        classifier=OpenAIResponsesDialectClassifier(
            model=args.model,
            confidence_mode=args.confidence_mode,
        ),
        train=[],
        test=test,
        output_dir=args.output,
        train_file=None,
        test_file=test_file,
        dataset_id="iadd_dart_region",
        task="dialect_region_identification",
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "model": report["model"],
                "examples": len(test),
                "accuracy": report["metrics"]["accuracy"],
                "macro_f1": report["metrics"]["macro_f1"],
                "sac_95": report["metrics"]["sac_95"]["coverage"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "cache_dir", False) is None:
        args.cache_dir = default_cache_dir()
    if args.command == "pilot":
        return run_massive_pilot(args)
    if args.command == "pilot-aratox":
        return run_aratox_pilot(args)
    if args.command == "audit":
        report = audit_normalized_splits(args.train, args.test)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "pilot-ollama-aratox":
        return run_ollama_aratox_pilot(args)
    if args.command == "pilot-aratox-multilabel":
        return run_aratox_multilabel_pilot(args)
    if args.command == "pilot-ollama-aratox-multilabel":
        return run_ollama_aratox_multilabel_pilot(args)
    if args.command == "pilot-ollama-iadd":
        return run_ollama_iadd_pilot(args)
    if args.command == "pilot-iadd-supervised":
        return run_iadd_supervised_pilot(args)
    if args.command == "pilot-openai-iadd":
        return run_openai_iadd_pilot(args)
    if args.command == "make-review-bundle":
        examples = list(read_multilabel_examples(args.examples))
        create_multilabel_review_tasks(
            examples=examples,
            output_path=args.manifest,
            seed=args.seed,
        )
        build_offline_review_bundle(
            task_manifest_path=args.manifest,
            template_path=args.template,
            output_path=args.bundle,
        )
        print(
            json.dumps(
                {
                    "tasks": len(examples),
                    "manifest": str(args.manifest),
                    "bundle": str(args.bundle),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "aggregate-reviews":
        aggregate = aggregate_submissions(
            task_manifest_path=args.manifest,
            submission_paths=args.submissions,
            output_path=args.output,
        )
        print(json.dumps(aggregate["summary"], indent=2))
        return 0
    if args.command == "build-release":
        payload = build_release(
            report_paths=args.reports,
            output_dir=args.output,
            release_id=args.release_id,
            comparison_paths=args.comparisons,
        )
        print(
            json.dumps(
                {
                    "release_id": payload["release_id"],
                    "result_count": payload["result_count"],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "compare-runs":
        comparison = compare_prediction_files(
            args.a,
            args.b,
            iterations=args.iterations,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2
