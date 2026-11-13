#!/usr/bin/env python3
"""Run Guess-the-SMILES on the same fresh models as the other tasks.

One seeded draw builds 100 molecules. Every model sees the same ASCII
drawing of each molecule. Scoring follows molecule_app.app.evaluate_prediction:
a reply is correct when its canonical SMILES equals the target, chemical
similarity is the Dice score of radius-2 Morgan fingerprints, and a reply
RDKit cannot canonicalize has similarity -1. The Flask app is not imported;
the runner and web app share molecule_app.scoring without server side effects.

Fable, Opus, Astra, and Sol are called once, with the same effort settings
as the board-game run. Luna is called at temperatures 0, 0.5, and 1.
Spend is charged against the caps already used by fresh_run.py.

Args:
    None. Configuration is the constants below and the flags parsed in main.

Returns:
    None. Writes gts_stimuli.json and gts_results.jsonl under
    fresh_experiments/seed<SEED>/.
"""

import argparse
import json
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import rdkit.Chem as rdc
import selfies as sf
from dotenv import load_dotenv
from rdkit import Chem, RDLogger

import fresh_run
from molecule_app.scoring import MAX_SMILES_LENGTH, score_prediction
from fresh_run import (
    CAPS,
    CONDITIONS,
    ROOT,
    SEED,
    _SPENT,
    _STOPPED,
    anthropic_limit_message,
    blocked,
    call_model,
)

RDLogger.DisableLog("rdApp.*")

N_MOLECULES = 100
LENGTH = 30
MIN_ATOMS = 10
MAX_ATOMS = 15
EXCLUDED = {"[=B]", "[#B]", "[=P]", "[#P]", "[#S]", "[Cl]", "[Br]", "[I]"}
OUT_DIR = ROOT / "fresh_experiments" / f"seed{SEED}"
STIMULI_PATH = OUT_DIR / "gts_stimuli.json"
LEDGER_PATH = OUT_DIR / "gts_results.jsonl"
SYSTEM = "You are a SMILES interpreter."
USER_TEMPLATE = (
    "The following is a depiction of a molecule that uses ASCII characters. "
    "The depiction uses element symbols for the corresponding atoms and * characters "
    "for single bonds between the atoms. Hence, the * characters are placed between "
    "two atoms that are connected by a single bond. Based on this depiction, write "
    "the SMILES string of the depicted molecule. Make sure that the SMILES string is "
    "correct and does not include the * characters:\n{ascii}\n"
    "Please provide the corresponding SMILES string, write nothing else but the SMILES string."
)
PREFIX = "The SMILES string for the given molecule is:"

_WRITE_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()


def _ascii_renderer():
    """Load the ASCII drawer from the molecule app without starting Flask.

    Returns:
        Callable: print_mol_ascii(mol) -> str.
    """
    app_dir = str(ROOT / "molecule_app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    from make_dataset import print_mol_ascii
    return print_mol_ascii


PRINT_MOL_ASCII = _ascii_renderer()


def alphabet():
    """Return the SELFIES tokens the Guess-the-SMILES generator keeps.

    Charged tokens and the unusual atom types named in the app are removed.
    Sort the set returned by selfies so new draws do not depend on Python's
    process hash seed. Existing frozen stimuli remain the canonical inputs
    for their saved runs.

    Returns:
        list[str]: SELFIES tokens.
    """
    tokens = sorted(sf.get_semantic_robust_alphabet())
    tokens = [token for token in tokens if "+" not in token and "-" not in token]
    return [token for token in tokens if token not in EXCLUDED]


def draw_cleaned_smiles(rng, tokens):
    """Draw one molecule and return the cleaned SMILES the API would store.

    The draw matches molecule_app.app.random_selfies: a SELFIES string of
    LENGTH tokens, an atom count inside the published bounds, a successful
    sanitize, then Kekule SMILES with '#' and '=' removed and no explicit H.
    The returned string is what later gets drawn and scored.

    Args:
        rng: Seeded random.Random. Only this generator is consumed.
        tokens: Alphabet from alphabet().

    Returns:
        str | None: Cleaned SMILES, or None when this draw is rejected.
    """
    selfie = "".join(rng.choices(tokens, k=LENGTH))
    mol = rdc.MolFromSmiles(sf.decoder(selfie))
    if mol is None:
        return None
    atoms = mol.GetNumAtoms(onlyExplicit=False)
    if atoms < MIN_ATOMS or atoms > MAX_ATOMS:
        return None
    if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
        return None
    smiles = rdc.MolToSmiles(mol, kekuleSmiles=True)
    cleaned = smiles.replace("#", "").replace("=", "")
    if "H" in cleaned:
        return None
    if rdc.MolFromSmiles(cleaned) is None:
        return None
    return cleaned


def build_stimuli(n, seed):
    """Build n molecules and their ASCII drawings from one seed.

    Args:
        n: Number of molecules.
        seed: Integer seed. The same seed rebuilds the same list.

    Returns:
        list[dict]: Entries with id, smiles, and ascii.
    """
    rng = random.Random(seed)
    tokens = alphabet()
    molecules = []
    for _ in range(max(10000, n * 10000)):
        if len(molecules) == n:
            break
        cleaned = draw_cleaned_smiles(rng, tokens)
        if cleaned is None:
            continue
        mol = rdc.MolFromSmiles(cleaned)
        molecules.append({
            "id": f"gts-{len(molecules)}",
            "smiles": cleaned,
            "ascii": PRINT_MOL_ASCII(mol),
        })
    if len(molecules) != n:
        raise RuntimeError(f'Only generated {len(molecules)}/{n} molecules within the draw budget')
    return molecules


def load_or_build(n, seed):
    """Return the frozen molecules, building them on the first call.

    Args:
        n: Number of molecules required.
        seed: Seed stored next to the molecules.

    Returns:
        list[dict]: The frozen molecule list.
    """
    if STIMULI_PATH.exists():
        payload = json.loads(STIMULI_PATH.read_text())
        if payload.get("seed") == seed and len(payload["molecules"]) >= n:
            return payload["molecules"][:n]
        raise RuntimeError(f"{STIMULI_PATH} was built with a different seed or size")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    molecules = build_stimuli(n, seed)
    STIMULI_PATH.write_text(json.dumps({"seed": seed, "alphabet": alphabet(), "molecules": molecules}))
    return molecules


def extract_smiles(response_text):
    """Take the SMILES span the existing harness takes.

    Args:
        response_text: Raw assistant text.

    Returns:
        str: Text after the fixed prefix, or the stripped reply.
    """
    if PREFIX in response_text:
        return response_text.split(PREFIX)[-1].strip()
    return response_text.strip()


def completion_ceiling(condition):
    """Return the completion-token ceiling for one condition.

    Luna does not spend tokens on reasoning. The other four models can,
    so they get a higher ceiling. The ceiling is still small enough that
    a full run stays inside the remaining dollar caps.

    Args:
        condition: Condition dict.

    Returns:
        int: max_tokens for this call.
    """
    if condition["model"] == "gpt-6-luna":
        return 512
    return 1024


def prior_spend():
    """Add the board-game spend and any Guess-the-SMILES spend already logged.

    Returns:
        dict[str, float]: Dollars by API model id.
    """
    spent = defaultdict(float)
    summary_path = OUT_DIR / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for model, amount in summary.get("spend_usd", {}).items():
            spent[model] += amount
    if LEDGER_PATH.exists():
        with LEDGER_PATH.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                spent[row["model"]] += row.get("cost_usd") or 0.0
    return spent


def done_keys():
    """Return episode keys that already have a ledger row.

    Returns:
        set[str]: Keys of the form condition|episode.
    """
    found = set()
    if not LEDGER_PATH.exists():
        return found
    with LEDGER_PATH.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            found.add(f"{row['condition']}|{row['episode']}")
    return found


def append_row(row):
    """Append one JSON object to the Guess-the-SMILES ledger.

    Args:
        row: Ledger object.

    Returns:
        None.
    """
    with _WRITE_LOCK:
        with LEDGER_PATH.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()


def play(condition, molecule):
    """Ask one model for one molecule and score the reply.

    Args:
        condition: Condition dict from fresh_run.CONDITIONS.
        molecule: Stimulus dict with id, smiles, and ascii.

    Returns:
        dict: Ledger row. error is "budget" when the cap blocks the call.
    """
    row = {
        "condition": condition["id"],
        "model": condition["model"],
        "temperature": condition["temperature"],
        "task": "gts",
        "episode": molecule["id"],
        "correct": False,
        "chemical_similarity": 0.0,
        "string_distance": -1,
        "invalid": False,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "error": None,
    }
    if blocked(condition["model"]):
        row["error"] = "budget"
        return row
    text, usage = call_model(
        condition,
        SYSTEM,
        USER_TEMPLATE.format(ascii=molecule["ascii"]),
        completion_ceiling(condition),
    )
    row["input_tokens"] = usage.get("input_tokens", 0) or 0
    row["output_tokens"] = usage.get("output_tokens", 0) or 0
    row["cost_usd"] = usage.get("cost_usd", 0.0) or 0.0
    predicted = extract_smiles(text)
    scored = score_prediction(molecule["smiles"], predicted)
    row.update(scored)
    row["response"] = text
    row["predicted"] = predicted
    row["truncated"] = row["output_tokens"] >= completion_ceiling(condition) - 1
    return row


def summarize():
    """Aggregate the Guess-the-SMILES ledger.

    Primary accuracy uses every reply, including invalid SMILES. The
    conditional accuracy excluding invalid SMILES is reported separately.
    Similarity is the mean chemical similarity, and invalid replies
    contribute -1.

    Returns:
        dict: Per condition, counts, accuracy, mean similarity, and spend.
    """
    buckets = defaultdict(lambda: {
        "n": 0, "correct": 0, "invalid": 0, "similarity": 0.0, "cost_usd": 0.0,
    })
    if LEDGER_PATH.exists():
        with LEDGER_PATH.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                bucket = buckets[row["condition"]]
                bucket["n"] += 1
                bucket["correct"] += int(bool(row.get("correct")))
                bucket["invalid"] += int(bool(row.get("invalid")))
                bucket["similarity"] += float(row.get("chemical_similarity") or 0.0)
                bucket["cost_usd"] += row.get("cost_usd") or 0.0
    report = {}
    for condition, bucket in sorted(buckets.items()):
        graded = bucket["n"] - bucket["invalid"]
        accuracy = 100.0 * bucket["correct"] / bucket['n'] if bucket['n'] else None
        report[condition] = {
            **bucket,
            "incorrect": bucket["n"] - bucket["correct"] - bucket["invalid"],
            "accuracy": accuracy,
            "conditional_accuracy": None if graded == 0 else 100.0 * bucket['correct'] / graded,
            "mean_similarity": bucket["similarity"] / bucket["n"] if bucket["n"] else None,
        }
    return report


def run(workers, limit):
    """Score every unfinished molecule.

    Args:
        workers: Concurrent calls.
        limit: If set, only the first this many molecules are used.

    Returns:
        dict: The summary of the ledger after the run.
    """
    load_dotenv(ROOT / ".env")
    molecules = load_or_build(N_MOLECULES, SEED)
    if limit is not None:
        molecules = molecules[:limit]
    conditions = CONDITIONS
    if any(condition["provider"] == "anthropic" for condition in CONDITIONS):
        limit_message = anthropic_limit_message()
        if limit_message:
            print(f"claude unavailable, skipping fable and opus: {limit_message}", flush=True)
            conditions = [condition for condition in CONDITIONS if condition["provider"] != "anthropic"]
    for model, amount in prior_spend().items():
        _SPENT[model] = amount
        if amount >= CAPS[model]:
            _STOPPED.add(model)
    finished = done_keys()
    jobs = []
    for condition in conditions:
        for molecule in molecules:
            if f"{condition['id']}|{molecule['id']}" not in finished:
                jobs.append((condition, molecule))
    print(f"molecules {len(molecules)} calls remaining {len(jobs)}", flush=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(play, condition, molecule) for condition, molecule in jobs]
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:
                print(f"call failed: {exc}", flush=True)
                continue
            append_row(row)
            completed += 1
            if completed % 20 == 0 or completed == len(jobs):
                report = summarize()
                spent = ", ".join(
                    f"{name} ${bucket['cost_usd']:.2f}" for name, bucket in sorted(report.items())
                )
                print(f"finished {completed}/{len(jobs)} gts-spend {spent}", flush=True)
    return summarize()


def self_test():
    """Check that scoring and the seeded draw do what the API does.

    Returns:
        None. Raises AssertionError when a check fails.
    """
    identity = score_prediction("CCO", "CCO")
    assert identity["correct"] is True
    assert identity["chemical_similarity"] == 1.0
    assert identity["string_distance"] == 0
    assert identity["invalid"] is False
    invalid = score_prediction("CCO", "this is not smiles")
    assert invalid["correct"] is False
    assert invalid["chemical_similarity"] == -1.0
    assert invalid["invalid"] is True
    empty = score_prediction("CCO", "")
    assert empty["chemical_similarity"] == 0.0
    assert empty["string_distance"] == -1
    assert empty["invalid"] is False
    long_reply = score_prediction("CCO", "C" * 201)
    assert long_reply["chemical_similarity"] == 0.0
    assert extract_smiles(f"prose. {PREFIX} CCO") == "CCO"
    first = build_stimuli(2, SEED)
    second = build_stimuli(2, SEED)
    assert [item["smiles"] for item in first] == [item["smiles"] for item in second]
    assert first[0]["ascii"] == second[0]["ascii"]
    assert "\n" in first[0]["ascii"]
    assert rdc.MolFromSmiles(first[0]["smiles"]) is not None
    print("self-test ok", flush=True)


def main():
    """Parse flags and either self-test, summarize, or run the benchmark.

    Returns:
        None.
    """
    global OUT_DIR, STIMULI_PATH, LEDGER_PATH, CONDITIONS
    parser = argparse.ArgumentParser(description="Fresh Guess-the-SMILES run.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, help="Separate directory for this run's ledger")
    parser.add_argument("--stimuli", type=Path, help="Existing frozen molecule JSON to reuse")
    parser.add_argument("--conditions", help="Comma-separated condition IDs for this run")
    args = parser.parse_args()
    if args.conditions:
        CONDITIONS = fresh_run.select_conditions(args.conditions)
    if args.out_dir:
        OUT_DIR = args.out_dir.resolve()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        STIMULI_PATH = OUT_DIR / "gts_stimuli.json"
        LEDGER_PATH = OUT_DIR / "gts_results.jsonl"
    if args.stimuli:
        STIMULI_PATH = args.stimuli.resolve()
    started = time.time()
    if args.self_test:
        self_test()
        return
    if args.summarize:
        print(json.dumps(summarize(), indent=2))
        return
    report = run(args.workers, args.limit)
    print(json.dumps(report, indent=2), flush=True)
    print(f"elapsed {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
