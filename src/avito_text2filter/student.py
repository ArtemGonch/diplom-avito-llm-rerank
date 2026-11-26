"""Train the same compact classifier with either reference labels or reference + Qwen labels."""

from collections import Counter
import hashlib
import random

from .study import SEED, family_key, filter_key


def description(row):
    value = row.get("value_name")
    if value is None:
        value = f"{row.get('min')} … {row.get('max')}"
    return f"{row['attr_name']} = {value} | {row['evidence']}"


def labeled_examples(catalog, propose, rows):
    examples = []
    missed = 0
    for row in rows:
        if not isinstance(row["filters"], list) or row.get("status") in {"pending", "disputed", "model_error"}:
            raise ValueError("Training requires reviewed labels")
        gold = {filter_key(f) for f in row["filters"]}
        _, options = propose(catalog, row["query"])
        keys = {filter_key(option) for option in options}
        missed += len(gold - keys)
        for option in options:
            examples.append((row["query"], description(option), int(filter_key(option) in gold)))
    return examples, missed


def training_batch(rng, reference, weak):
    """Keep half of each label's batch tied to the reference when weak pairs exist."""
    examples = []
    for label in (0, 1):
        if reference[label] and weak[label]:
            examples.extend(rng.choices(reference[label], k=4))
            examples.extend(rng.choices(weak[label], k=4))
        else:
            examples.extend(rng.choices(reference[label] or weak[label], k=8))
    return examples


def train(catalog, propose, args, read_rows):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    by_id = {r["id"]: r for r in read_rows(args.manifest)}
    human = read_rows(args.train)
    expected = {key for key, row in by_id.items() if row["split"] == "train"}
    if not expected or len(human) != len(expected) or {r["id"] for r in human} != expected:
        raise ValueError("The reference training set must match the frozen train split")
    if any(hashlib.sha256(r["query"].encode()).hexdigest() != by_id[r["id"]]["query_sha256"]
           for r in human):
        raise ValueError("Reference training query differs from its frozen family")
    pseudo = read_rows(args.pseudo) if args.pseudo else []
    if args.pseudo and not pseudo:
        raise ValueError("Expected reviewed model labels")
    reserved = {row["family_key"] for row in by_id.values()}
    pseudo_keys = {family_key(row["query"]) for row in pseudo}
    if pseudo_keys & reserved or len(pseudo_keys) != len(pseudo):
        raise ValueError("Model labels must be unique and outside the reference packet")
    eligible = {"reviewed"}
    if args.linker == "v2":
        eligible.add("qwen_consensus")
    if any(row.get("status") not in eligible for row in pseudo):
        raise ValueError("Expected audited labels or two agreeing Qwen selections")
    reference_examples, reference_missed = labeled_examples(catalog, propose, human)
    weak_examples, weak_missed = labeled_examples(catalog, propose, pseudo)
    examples = reference_examples + weak_examples
    counts = Counter(c for _, _, c in examples)
    if set(counts) != {0, 1} or args.steps < 1:
        raise ValueError("Training needs both positive and negative linked examples and at least one step")

    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=2, local_files_only=True
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    reference = {label: [ex for ex in reference_examples if ex[2] == label] for label in (0, 1)}
    weak = {label: [ex for ex in weak_examples if ex[2] == label] for label in (0, 1)}
    if args.linker == "v1":
        reference = {label: reference[label] + weak[label] for label in (0, 1)}
        weak = {0: [], 1: []}
    for _ in range(args.steps):
        batch = training_batch(rng, reference, weak)
        left, right, targets = zip(*batch, strict=True)
        inputs = tokenizer(list(left), list(right), padding=True, truncation=True,
                           max_length=128, return_tensors="pt").to(device)
        loss = model(**inputs, labels=torch.tensor(targets, device=device)).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    args.out.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    return {"reference_queries": len(human), "pseudo_queries": len(pseudo), "steps": args.steps,
            "examples": dict(counts),
            "reference_pairs": dict(Counter(c for _, _, c in reference_examples)),
            "weak_pairs": dict(Counter(c for _, _, c in weak_examples)),
            "gold_filters_outside_options": reference_missed + weak_missed, "device": device}


def student_runner(catalog, model_path, propose, assemble, threshold=0.5, allow_linked=False):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).eval()

    def infer(query):
        baseline, options = propose(catalog, query)
        if not options:
            return baseline
        inputs = tokenizer([query] * len(options), [description(row) for row in options],
                           padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.inference_mode():
            probabilities = model(**inputs).logits.softmax(dim=-1)[:, 1].tolist()
        ranked = sorted(zip(probabilities, options, strict=True), key=lambda item: -item[0])
        choices = []
        attributes, spans = set(), {}
        for probability, row in ranked:
            attr, span = row["attr_id"], (row["start"], row["end"])
            if (probability >= threshold and attr not in attributes
                    and (span not in spans or (allow_linked
                         and {spans[span], attr} == {"110000", "110001"}))):
                attributes.add(attr)
                spans[span] = attr
                choices.append(row)
        return assemble(baseline, choices, allow_linked=allow_linked)
    return infer
