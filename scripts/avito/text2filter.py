#!/usr/bin/env python3
"""Local CLI and demo for UsedCars query-to-filter extraction."""

import argparse
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import random
import sys
from threading import Lock
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from avito_text2filter.study import (
    SEED, assemble, build_packet, consensus_choices, ensure_disjoint_queries, evaluate, family_key, filter_key, latency_summary,
    paired_exact_filters, selected_options,
)


def save_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as file:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), file=file)


def read_rows(path):
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as file:
            result = list(csv.DictReader(file))
        for row in result:
            for field in ("filters", "unresolved"):
                row[field] = json.loads(row[field]) if row[field].strip() else None
        return result
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def packet(archive_path, output):
    with ZipFile(archive_path) as archive:
        manifest, splits = build_packet(archive)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    save_jsonl(output / "sampling.jsonl", manifest)
    summary = {key: len(rows) for key, rows in splits.items()}
    for split, rows in splits.items():
        with (output / f"{split}.csv").open("x", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    with archive_path.open("rb") as file:
        archive_hash = hashlib.file_digest(file, "sha256").hexdigest()
    (output / "manifest.json").write_text(json.dumps({
        "source_sha256": archive_hash,
        "sampling_sha256": hashlib.sha256((output / "sampling.jsonl").read_bytes()).hexdigest(),
        "seed": SEED, "counts": summary,
    }, indent=2) + "\n", encoding="utf-8")
    return summary


def proposals(catalog, query):
    baseline = catalog.parse(query)
    options = baseline["filters"] + catalog.candidates(query)
    return baseline, options[:20]

def _unique_options(rows):
    identities = [(filter_key(row), row["start"], row["end"]) for row in rows]
    return list(dict(zip(identities, rows, strict=True)).values())[:20]


def proposals_v2(catalog, query):
    baseline = catalog.parse(query)
    return baseline, _unique_options(baseline["filters"] + catalog.model_candidates(query, baseline))


def proposals_v3(catalog, query):
    baseline = catalog.parse(query)
    return baseline, _unique_options(baseline["filters"] + catalog.model_candidates_v3(query, baseline))


def proposals_v4(catalog, query):
    baseline = catalog.baseline_v4(query, catalog.parse(query))
    return baseline, _unique_options(baseline["filters"] + catalog.model_candidates_v4(query, baseline))


_LEGACY_PROMPT = (
    'Выбери явно запрошенные автомобильные фильтры из списка. '
    'Подписи из запроса могут быть шумом. Ответ строго JSON: {"accept":[0,2]}. '
    'Индексы только из списка, без пояснений. Если нет подходящих — {"accept":[]}.'
)
_TEACHER_PROMPTS = (
    'Размечай запрос для поиска подержанных автомобилей. Прими варианты, которые выражают '
    'явное требование к самому автомобилю. Если ищут деталь, ремонт, игрушку, аренду, обмен '
    'запчастей или аксессуар, оставь пустой список. Название машины в запросе запчасти '
    'лишь указывает совместимость. Сохрани корректно распознанные марки, модели, единицы и границы. '
    'Если в тексте названа модель, прими её однозначно связанную марку, когда она есть в списке. '
    'Не превращай исключение или пожелание вне справочника в положительный фильтр.',
    'Проверь независимо Применимые фильтры в поиске авто с пробегом. Для каждого номера '
    'реши, ищет ли человек сам автомобиль и доказывает ли приведённый фрагмент именно данное значение. '
    'Примеры: "фары для Ford" → ни одного; "ищу Ford до 2 млн" → марка и максимум цены; '
    '"не дороже 500 тыс" → максимум цены; "Tesla без ДТП" → марка. '
    'Пропусти вариант, если число относится к версии модели или перечислен неподдерживаемый выбор.',
)


def qwen_runner(catalog, model_path, propose=proposals, consensus=False):
    allow_linked = propose in (proposals_v2, proposals_v3, proposals_v4)
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(str(model_path))
    sampler = make_sampler(temp=0.0)

    def request(query, options, system):
        payload = [[i, row["attr_name"], row.get("value_name"), row.get("min"),
                    row.get("max"), row["evidence"]] for i, row in enumerate(options)]
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"query": query, "options": payload}, ensure_ascii=False)},
        ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return generate(model, tokenizer, prompt=prompt, max_tokens=64, sampler=sampler).strip()

    def infer(query):
        baseline, options = propose(catalog, query)
        if not options:
            return ({**baseline, "status": "no_candidates"} if consensus else baseline)
        try:
            if not consensus:
                raw = request(query, options, _LEGACY_PROMPT)
                return assemble(baseline, selected_options(raw, options), allow_linked=allow_linked)
            suffix = ' Верни только JSON объект {"accept":[0,2]} с выбранными индексами или {"accept":[]}.'
            direct = request(query, options, _TEACHER_PROMPTS[0] + suffix)
            reverse = request(query, options[::-1], _TEACHER_PROMPTS[1] + suffix)
            choices = consensus_choices(direct, reverse, options)
            if choices is None:
                return {**baseline, "status": "qwen_disagreement"}
            return {**assemble(baseline, choices, allow_linked=allow_linked), "status": "qwen_consensus"}
        except ValueError:
            return {**baseline, "status": "model_error"}
    return infer

def run_rows(rows, infer):
    for row in rows:
        start = time.perf_counter()
        result = infer(row["query"])
        yield {"id": row["id"], "query": row["query"], **result,
               "latency_ms": (time.perf_counter() - start) * 1000}


def training_pool(archive_path, manifest, limit):
    reserved = {row["family_key"] for row in manifest}
    with ZipFile(archive_path) as archive, io.TextIOWrapper(
        archive.open("query_frequencies.csv"), encoding="utf-8", newline=""
    ) as file:
        available = {}
        for row in csv.DictReader(file):
            query = row["query_text"]
            key = family_key(query)
            if key not in reserved:
                available.setdefault(key, query)
    head = list(available.values())[:max(2 * limit, 200)]
    tail = list(available.values())[max(2 * limit, 200):]
    rng = random.Random(SEED)
    return [{"id": f"pseudo-{i:05}", "query": q}
            for i, q in enumerate(rng.sample(head, (limit + 1) // 2) + rng.sample(tail, limit // 2))]


_DEMO_EXAMPLES = (
    "Toyota Camry 2.5 автомат без ДТП",
    "BMW 528i 2014 г до 1,5 млн",
    "не дизель, от 2020 года",
)
_DEMO_COMPARISON = {"baseline": "v2_control", "candidate": "v3_verified_qwen_calibrated"}
_DEMO_METHODS = (
    ("rules", "Правила"),
    ("v2_control", "BERT v2 · бейзлайн"),
    ("v3_verified_qwen_calibrated", "BERT v3 · Astra + Qwen"),
    ("v4_frozen_weights", "v4 · веса v3"),
    ("v4_refit", "v4 · переобучение"),
)


def demo_page(catalog, *, student=None, student_label=None, offline=False):
    aggregate = ROOT / "results/current/metrics/avito_text2filter_v4_20261126.json"
    result = json.loads(aggregate.read_text(encoding="utf-8"))
    bench = []
    for method, label in _DEMO_METHODS:
        metrics = result["metrics"][method]["families"]
        bench.append({
            "method": method, "label": label, "raw_exact": result["raw_exact"][method],
            **{key: metrics[key] for key in ("exact_filters", "filter_precision", "filter_recall")},
        })
    config = {
        "examples": _DEMO_EXAMPLES, "student_label": student_label if student is not None else None,
        "offline": offline,
        "benchmark": {"queries": result["design"]["test_rows"], "rows": bench, "comparison": _DEMO_COMPARISON},
    }
    if offline:
        replays = {query: {"rules": catalog.parse(query)} for query in _DEMO_EXAMPLES}
        if student is not None:
            for query, replay in replays.items():
                replay["bert"] = student(query)
        config["replays"] = replays
    template = Path(__file__).with_name("text2filter_demo.html").read_text(encoding="utf-8")
    payload = json.dumps(config, ensure_ascii=False, allow_nan=False).replace("<", r"\u003c")
    return template.replace("__DEMO_CONFIGURATION__", payload, 1).encode()


def serve(catalog, port, *, student=None, student_label=None):
    page = demo_page(catalog, student=student, student_label=student_label)
    student_lock = Lock()

    class DemoHandler(BaseHTTPRequestHandler):
        def respond(self, status, content_type, body):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.respond(200, "text/html; charset=utf-8", page)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/parse" and (self.path != "/parse/bert" or student is None):
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16_384:
                    raise ValueError("Invalid payload size")
                query = json.loads(self.rfile.read(size))["query"]
                if not isinstance(query, str) or len(query) > 1000:
                    raise ValueError("Invalid query")
            except (ValueError, KeyError, TypeError):
                self.send_error(400)
                return
            if self.path == "/parse/bert":
                with student_lock:
                    result = student(query)
            else:
                result = catalog.parse(query)
            body = json.dumps(result, ensure_ascii=False).encode()
            self.respond(200, "application/json; charset=utf-8", body)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), DemoHandler) as server:
        print(f"http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--linker", choices=("v1", "v2", "v3", "v4"), default="v1")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("parse")
    p.add_argument("query")
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=8782)
    p.add_argument("--student", type=Path, help="Optional local BERT checkpoint for the demo")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--export", type=Path, help="Write a standalone HTML with saved example outputs, then exit")
    p = sub.add_parser("packet")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("predict")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    models = p.add_mutually_exclusive_group()
    models.add_argument("--qwen", type=Path)
    models.add_argument("--student", type=Path)
    p.add_argument("--threshold", type=float, default=0.5)
    p = sub.add_parser("teacher")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--limit", type=int, default=50, choices=range(1, 2001), metavar="1..2000")
    p.add_argument("--consensus", action="store_true")
    p.add_argument("--input", type=Path, help="Preselected unlabeled training queries")
    p.add_argument("--qwen", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("train")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--pseudo", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=300)
    p = sub.add_parser("score")
    for name in ("gold", "predictions", "manifest"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p = sub.add_parser("compare")
    for name in ("gold", "control", "augmented", "manifest"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "score":
        output = evaluate(read_rows(args.gold), read_rows(args.predictions), read_rows(args.manifest))
    elif args.command == "compare":
        gold, manifest = read_rows(args.gold), read_rows(args.manifest)
        control, augmented = read_rows(args.control), read_rows(args.augmented)
        output = {"control": evaluate(gold, control, manifest),
                  "augmented": evaluate(gold, augmented, manifest),
                  "paired_exact_filters": paired_exact_filters(gold, control, augmented, manifest)}
    elif not args.archive:
        parser.error("--archive is required for this command")
    elif args.command == "packet":
        output = packet(args.archive, args.out)
    else:
        from avito_text2filter import Catalog
        catalog = Catalog.from_zip(args.archive)
        propose = {"v1": proposals, "v2": proposals_v2, "v3": proposals_v3, "v4": proposals_v4}[args.linker]
        if args.command == "serve":
            student = student_label = None
            if args.student:
                from avito_text2filter.student import student_runner
                if not 0 <= args.threshold <= 1:
                    parser.error("--threshold must be in [0, 1]")
                student = student_runner(catalog, args.student, propose, assemble, args.threshold,
                                         allow_linked=args.linker != "v1")
                student_label = f"BERT {args.linker}"
            if args.export:
                page = demo_page(catalog, student=student, student_label=student_label, offline=True)
                with args.export.open("xb") as file:
                    file.write(page)
            else:
                serve(catalog, args.port, student=student, student_label=student_label)
            return
        if args.command == "parse":
            output = catalog.parse(args.query)
        elif args.command == "train":
            from avito_text2filter.student import train
            output = train(catalog, propose, args, read_rows)
        else:
            if args.command == "teacher":
                manifest = read_rows(args.manifest)
                if args.input:
                    rows = read_rows(args.input)
                    ensure_disjoint_queries(rows, manifest)
                else:
                    rows = training_pool(args.archive, manifest, args.limit)
            else:
                rows = read_rows(args.input)
            infer = catalog.parse
            if args.qwen:
                infer = qwen_runner(catalog, args.qwen, propose,
                                    consensus=args.command == "teacher" and args.consensus)
            elif args.command == "predict" and args.student:
                from avito_text2filter.student import student_runner
                if not 0 <= args.threshold <= 1:
                    parser.error("--threshold must be in [0, 1]")
                infer = student_runner(catalog, args.student, propose, assemble, args.threshold,
                                       allow_linked=args.linker != "v1")
            if not rows:
                raise ValueError("No input queries")
            infer("автомат до 1,5 млн без ДТП")  # warm up outside the timed sample
            results = list(run_rows(rows, infer))
            save_jsonl(args.out, results)
            output = {"queries": len(results), "model_errors": sum(
                row.get("status") == "model_error" for row in results),
                "latency_ms": latency_summary([row["latency_ms"] for row in results])}
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
