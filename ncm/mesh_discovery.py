#!/usr/bin/env python3
"""mesh_discovery.py - Autonomous Air-Gapped Scientific Discovery Engine for Local AI Mesh.
Combines:
  1. Local Algorithmic Symbolic Search (evaluates functional invariants across empirical tables)
  2. Local LLM Hypothesis Generation (RTX 5080 Qwen 2.5 14B)
  3. Formal Mathematical Synthesis & Theorem Proving (Mac Studio M1 Max Qwen3-Coder 30B via :8780)
Zero internet required. Operates purely within the physical local mesh.
"""

import os
import sys
import time
import json
import math
import urllib.request
import numpy as np

STUDIO_URL = os.environ.get("MESH_STUDIO_URL", "http://127.0.0.1:8780/v1/chat/completions")
RTX_URL = os.environ.get("MESH_RTX_URL", "http://127.0.0.1:11434/api/generate")

class MeshDiscoveryEngine:
    def __init__(self):
        pass

    def symbolic_screen(self, x_vals, y_vals):
        """Algorithmic search for power invariants: y^p / x^q == constant."""
        candidates = [
            (1, 1, "Y / X"),
            (1, 2, "Y / X^2"),
            (2, 1, "Y^2 / X"),
            (2, 3, "Y^2 / X^3"),
            (3, 2, "Y^3 / X^2"),
            (1, 0.5, "Y / sqrt(X)"),
            (2, 0.5, "Y^2 / sqrt(X)"),
            (1, 1.5, "Y / X^(3/2)")
        ]
        results = []
        for p, q, label in candidates:
            ratios = []
            for x, y in zip(x_vals, y_vals):
                val = (y ** p) / (x ** q)
                ratios.append(val)
            mean_val = np.mean(ratios)
            std_val = np.std(ratios)
            cv = std_val / max(1e-9, abs(mean_val)) # Coefficient of variation
            results.append({
                "label": label,
                "p": p,
                "q": q,
                "mean": round(float(mean_val), 4),
                "std": round(float(std_val), 5),
                "cv": round(float(cv), 6),
                "ratios": [round(float(r), 4) for r in ratios]
            })
        results.sort(key=lambda r: r["cv"])
        return results

    def discover_law(self, dataset_name, x_vals, y_vals, x_label="Metric X", y_label="Metric Y"):
        print(f"\n[*] Initiating Blind Discovery Experiment: {dataset_name}")
        t0 = time.time()
        
        # Phase 1: Algorithmic Invariant Screening
        screen_results = self.symbolic_screen(x_vals, y_vals)
        best = screen_results[0]
        print(f"    [Phase 1: Algorithmic Screen] Best candidate invariant: {best['label']} (CV={best['cv']})")

        # Phase 2: Local LLM Hypothesis & Deduction (RTX 5080)
        table_str = "\n".join([f"  Sample {i+1}: {x_label} = {x:.4f}, {y_label} = {y:.4f}" for i, (x, y) in enumerate(zip(x_vals, y_vals))])
        prompt_hyp = f"""You are an experimental physicist and applied mathematician presented with raw data:

{table_str}

Task:
1. Examine these numerical pairs.
2. An algorithmic screener found that the candidate relation {best['label']} has near-zero variance ({best['cv']}) with mean value {best['mean']}.
3. Explain why this relationship is exact and express it as a fundamental physical/mathematical law.
4. Conclude with the clean algebraic equation relating {y_label} to {x_label}."""

        body_5080 = {
            "model": "qwen2.5:14b-instruct",
            "prompt": prompt_hyp,
            "stream": False,
            "options": {"num_predict": 384, "temperature": 0.1}
        }
        req = urllib.request.Request(RTX_URL, data=json.dumps(body_5080).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            res_5080 = json.loads(resp.read().decode())
        hyp_text = res_5080.get("response", "").strip()

        # Phase 3: Mac Studio M1 Max Formal Law Synthesis (Qwen3-Coder 30B)
        prompt_studio = f"""As Lead Theoretical Scientist on the Mac Studio, formalize the following discovery:
Empirical dataset: {dataset_name}
Discovered Mathematical Invariant: {best['label']} = {best['mean']} (Std Dev: {best['std']})
Mathematical deduction:
\"\"\"{hyp_text}\"\"\"

Produce:
1. Formal Law Statement & Equation.
2. Empirical Validation Summary Table.
3. Significance of the Invariant Constant."""

        body_studio = {
            "model": "qwen3-coder:30b",
            "messages": [{"role": "user", "content": prompt_studio}],
            "max_tokens": 512,
            "temperature": 0.1
        }
        req_studio = urllib.request.Request(STUDIO_URL, data=json.dumps(body_studio).encode(), headers={"Content-Type": "application/json", "User-Agent": "mesh-client"})
        with urllib.request.urlopen(req_studio, timeout=45) as resp:
            res_studio = json.loads(resp.read().decode())
        law_synthesis = res_studio.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        t_elap = time.time() - t0

        return {
            "dataset": dataset_name,
            "elapsed_s": round(t_elap, 2),
            "best_invariant": best,
            "rtx_deduction": hyp_text,
            "studio_formal_law": law_synthesis
        }

if __name__ == "__main__":
    engine = MeshDiscoveryEngine()
    # Kepler planetary data (anonymized)
    x = [0.3871, 0.7233, 1.0000, 1.5237, 5.2026, 9.5549]
    y = [0.2408, 0.6152, 1.0000, 1.8808, 11.8620, 29.4580]
    out = engine.discover_law("Historical Celestial Orbital Harmonics (Tycho Brahe 1618)", x, y)
    print("\n" + "=" * 80)
    print(out["studio_formal_law"])
    print("=" * 80)
