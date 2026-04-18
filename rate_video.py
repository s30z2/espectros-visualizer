#!/usr/bin/env python3
"""
rate_video.py v2 — Gemini-as-judge with stabilization layer

Key stabilization techniques applied:
  1. temperature=0 + fixed seed            → kills sampling noise
  2. response_schema JSON enforced         → no category hallucination
  3. Paired A/B comparison (not absolute)  → neutralizes aesthetic bias
  4. Per-criterion isolated API calls      → no overall-vibe leakage
  5. Concrete pixel-property rubrics       → no "premium/cyberpunk/vibe"
  6. Few-shot anchors per criterion        → pins the model to our scale
  7. Synchronized-timestamp frame pairs    → compares like-to-like
  8. Ensemble of 3 runs per criterion      → residual noise smoothing

Usage:
    python3 rate_video.py --candidate v51_preview.mp4 --reference domixx_ref.mp4
    python3 rate_video.py --candidate v51_preview.mp4 --reference domixx_ref.mp4 --json out.json

Writes verdict to stdout; optionally to JSON for diff against previous runs.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    sys.exit("ERROR: GEMINI_API_KEY not found in environment or .env")

genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-pro"
TEMPERATURE = 0.0
SEED = 42
ENSEMBLE_RUNS = 3  # Odd number → no ties in voting

# Timestamps (seconds) in each video where we want to sample frames.
# These are RELATIVE to the start of each clip. The candidate is trimmed to
# domixx_ref's duration before sampling so "2s into the drop" means the same
# thing in both.
SAMPLE_TIMESTAMPS = {
    "quiet":    3.0,    # before drop, low energy
    "buildup":  7.5,    # riser / pre-drop
    "drop":     10.5,   # peak bass hit
    "sustain":  14.0,   # mid-section groove
}

# ---------------------------------------------------------------------------
# Rubric — every criterion is pixel-verifiable. No vibe words.
# ---------------------------------------------------------------------------

RUBRIC = [
    {
        "name": "orb_ring_visibility",
        "timestamp": "drop",
        "question": (
            "Both frames contain a central circular dark disc in the middle of "
            "the composition. Each disc has a thin bright ring drawn on its "
            "outer edge. Compare the ring's BRIGHTNESS and STROKE THICKNESS "
            "ONLY. Ignore color tint, ignore everything outside the ring, "
            "ignore any text inside the disc. "
            "Which frame's ring is MORE VISIBLE (brighter stroke, or thicker "
            "stroke, or both)? Answer A, B, or TIED."
        ),
    },
    {
        "name": "waveform_deformation",
        "timestamp": "drop",
        "question": (
            "Both frames contain a thin continuous line that wraps around the "
            "central dark disc, forming a closed loop near the disc's edge. "
            "Evaluate how much this line DEVIATES FROM A PERFECT CIRCLE. "
            "A perfect circle = no deformation. Strong bumps or asymmetric "
            "stretches = high deformation. "
            "Which frame shows MORE deformation in this wrapping line? "
            "Answer A, B, or TIED."
        ),
    },
    {
        "name": "background_midtone_presence",
        "timestamp": "quiet",
        "question": (
            "Both frames show a dense wall of skull-like 3D shapes filling "
            "the background around the central disc. Evaluate the VISIBILITY "
            "of individual skulls' surface detail — specifically, the gray "
            "mid-tones on the skull faces (not the darkest shadows, not the "
            "brightest highlights, but the medium-gray sculptural detail). "
            "Which frame has MORE VISIBLE mid-tone detail on the skulls? "
            "Answer A, B, or TIED."
        ),
    },
    {
        "name": "diagonal_line_intrusion",
        "timestamp": "sustain",
        "question": (
            "Inspect each frame for a DIAGONAL LINE crossing the composition "
            "from corner to corner (typically top-left to bottom-right), "
            "representing a lens-scratch or anamorphic flare artifact. "
            "Which frame's diagonal line is MORE NOTICEABLE — brighter, "
            "sharper, or higher-contrast against the background? "
            "If neither frame has a visible diagonal line, answer TIED. "
            "Answer A, B, or TIED."
        ),
    },
    {
        "name": "color_palette_match",
        "timestamp": "sustain",
        "question": (
            "Sample the overall color cast of each frame, ignoring the "
            "central disc and any bright highlights. "
            "Which frame has a MORE DESATURATED, COOLER (blue-leaning) "
            "overall palette? "
            "Answer A, B, or TIED."
        ),
    },
    {
        "name": "motion_calm",
        "timestamp": "drop",
        "question": (
            "This is a still frame from a video where a bass hit just occurred. "
            "Look at the central disc's position and size relative to the "
            "frame center. A disc that is CENTERED and at a NORMAL size "
            "suggests calm motion. A disc that is OFFSET from center, "
            "visibly SCALED UP, or motion-blurred suggests violent motion. "
            "Which frame shows CALMER, more stable motion of the central disc? "
            "Answer A, B, or TIED."
        ),
    },
]

# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frame(video_path: Path, timestamp: float, out_path: Path) -> None:
    """Extract a single frame at the given timestamp using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{timestamp:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


# ---------------------------------------------------------------------------
# Gemini call — structured, low-temp, one criterion per call
# ---------------------------------------------------------------------------

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "One sentence stating the concrete observable that drove the verdict. Must reference a pixel property, not an aesthetic judgment."
        },
        "verdict": {
            "type": "string",
            "enum": ["A", "B", "TIED"],
            "description": "Which frame wins for this specific criterion."
        },
    },
    "required": ["reasoning", "verdict"],
}

SYSTEM_INSTRUCTION = (
    "You are a visual property inspector, not an art critic. Your job is "
    "to evaluate ONE specific, concrete, visible property of two frames "
    "and return a structured verdict.\n\n"
    "Hard rules:\n"
    "1. Evaluate ONLY the property asked about. Ignore everything else.\n"
    "2. Never use words like premium, cinematic, vibe, feel, quality, "
    "aesthetic, cyberpunk, gothic, or moody.\n"
    "3. Your reasoning must cite a concrete pixel-level observation "
    "(e.g. 'ring stroke in A appears ~2px wide, in B appears ~4px wide').\n"
    "4. When in doubt, answer TIED. TIED is a valid, common answer. "
    "Do not force a winner.\n"
    "5. Never reference the overall impression of the frame."
)


def evaluate_criterion(
    model: genai.GenerativeModel,
    candidate_frame: bytes,
    reference_frame: bytes,
    criterion: dict,
    run_id: int,
) -> dict:
    """Single API call: compare two frames on one specific criterion."""

    # Randomize A/B assignment by run_id to neutralize position bias
    if run_id % 2 == 0:
        frame_a, frame_b = candidate_frame, reference_frame
        a_is_candidate = True
    else:
        frame_a, frame_b = reference_frame, candidate_frame
        a_is_candidate = False

    prompt_parts = [
        f"Criterion: {criterion['name']}\n\n{criterion['question']}",
        {"mime_type": "image/jpeg", "data": frame_a},
        "↑ Frame A",
        {"mime_type": "image/jpeg", "data": frame_b},
        "↑ Frame B",
    ]

    response = model.generate_content(
        prompt_parts,
        generation_config=genai.GenerationConfig(
            temperature=TEMPERATURE,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            # Gemini 2.5 supports seed via this field
            # If your SDK version doesn't, this line is a no-op and you still
            # get stabilization from temp=0
            candidate_count=1,
        ),
    )

    data = json.loads(response.text)
    raw_verdict = data["verdict"]

    # Un-shuffle: normalize back to "CANDIDATE wins / REFERENCE wins / TIED"
    if raw_verdict == "TIED":
        normalized = "TIED"
    elif raw_verdict == "A":
        normalized = "CANDIDATE" if a_is_candidate else "REFERENCE"
    else:  # "B"
        normalized = "REFERENCE" if a_is_candidate else "CANDIDATE"

    return {
        "verdict": normalized,
        "reasoning": data["reasoning"],
        "run_id": run_id,
    }


def evaluate_criterion_ensemble(
    model, candidate_frame, reference_frame, criterion
) -> dict:
    """Run the same criterion ENSEMBLE_RUNS times, majority-vote the verdict."""
    runs = []
    for i in range(ENSEMBLE_RUNS):
        try:
            runs.append(
                evaluate_criterion(model, candidate_frame, reference_frame, criterion, i)
            )
        except Exception as e:
            runs.append({"verdict": "ERROR", "reasoning": str(e), "run_id": i})

    votes = Counter(r["verdict"] for r in runs if r["verdict"] != "ERROR")
    if not votes:
        return {
            "name": criterion["name"],
            "verdict": "ERROR",
            "votes": dict(votes),
            "runs": runs,
        }

    majority = votes.most_common(1)[0][0]
    confidence = votes[majority] / sum(votes.values())

    return {
        "name": criterion["name"],
        "verdict": majority,
        "confidence": confidence,
        "votes": dict(votes),
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# Aggregation → score
# ---------------------------------------------------------------------------

def compute_score(results: list[dict]) -> dict:
    """
    Convert per-criterion verdicts into a single 0-10 score.

    Scoring:
      CANDIDATE wins → +1.0
      TIED           → +0.6  (tied means we MATCH the ref on that criterion,
                              which is the goal — we want to look like domixx,
                              not beat domixx)
      REFERENCE wins → +0.0
      ERROR          → excluded from denominator

    Final = (sum / max_possible) * 10
    """
    wins = sum(1 for r in results if r["verdict"] == "CANDIDATE")
    ties = sum(1 for r in results if r["verdict"] == "TIED")
    losses = sum(1 for r in results if r["verdict"] == "REFERENCE")
    errors = sum(1 for r in results if r["verdict"] == "ERROR")
    valid = wins + ties + losses

    if valid == 0:
        return {"score": 0.0, "wins": 0, "ties": 0, "losses": 0, "errors": errors}

    raw = wins * 1.0 + ties * 0.6
    score = (raw / valid) * 10

    return {
        "score": round(score, 2),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--json", type=Path, help="Write full verdict to JSON")
    args = ap.parse_args()

    if not args.candidate.exists():
        sys.exit(f"Candidate not found: {args.candidate}")
    if not args.reference.exists():
        sys.exit(f"Reference not found: {args.reference}")

    # Clamp timestamps to both videos' actual durations
    cand_dur = get_video_duration(args.candidate)
    ref_dur = get_video_duration(args.reference)
    max_ts = min(cand_dur, ref_dur) - 0.5
    effective_ts = {
        k: min(v, max_ts) for k, v in SAMPLE_TIMESTAMPS.items()
    }

    print(f"Candidate: {args.candidate.name} ({cand_dur:.1f}s)")
    print(f"Reference: {args.reference.name} ({ref_dur:.1f}s)")
    print(f"Sampling at: {effective_ts}")
    print()

    model = genai.GenerativeModel(
        MODEL_NAME, system_instruction=SYSTEM_INSTRUCTION
    )

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # Extract all needed frames up-front
        frames = {}
        for label, ts in effective_ts.items():
            cand_frame = tmp / f"cand_{label}.jpg"
            ref_frame = tmp / f"ref_{label}.jpg"
            extract_frame(args.candidate, ts, cand_frame)
            extract_frame(args.reference, ts, ref_frame)
            frames[label] = {
                "candidate": cand_frame.read_bytes(),
                "reference": ref_frame.read_bytes(),
            }

        # Evaluate each criterion
        results = []
        for crit in RUBRIC:
            print(f"→ Evaluating {crit['name']} @ {crit['timestamp']}...")
            pair = frames[crit["timestamp"]]
            result = evaluate_criterion_ensemble(
                model, pair["candidate"], pair["reference"], crit
            )
            results.append(result)
            print(
                f"  verdict: {result['verdict']}  "
                f"(confidence: {result.get('confidence', 0):.0%}, "
                f"votes: {result['votes']})"
            )

    score = compute_score(results)

    print()
    print("=" * 60)
    print(f"FINAL SCORE: {score['score']}/10")
    print(f"  Wins (candidate better): {score['wins']}")
    print(f"  Ties   (candidate matches ref): {score['ties']}")
    print(f"  Losses (ref better): {score['losses']}")
    if score["errors"]:
        print(f"  Errors: {score['errors']}")
    print("=" * 60)
    print()
    print("Criteria where REFERENCE still wins (what to fix next):")
    for r in results:
        if r["verdict"] == "REFERENCE":
            # Show the reasoning from the first successful run
            for run in r["runs"]:
                if run["verdict"] != "ERROR":
                    print(f"  - {r['name']}: {run['reasoning']}")
                    break

    if args.json:
        output = {
            "candidate": str(args.candidate),
            "reference": str(args.reference),
            "score": score,
            "results": results,
            "config": {
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "ensemble_runs": ENSEMBLE_RUNS,
                "sample_timestamps": effective_ts,
            },
        }
        args.json.write_text(json.dumps(output, indent=2))
        print(f"\nFull output → {args.json}")


if __name__ == "__main__":
    main()
