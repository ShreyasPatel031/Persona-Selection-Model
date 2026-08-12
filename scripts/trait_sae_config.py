"""Shared trait → run_id / layer / SAE mapping for OMP decomposition pipeline."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SAE_RELEASE = "gemma-scope-2-4b-it-res-all"
DEFAULT_ALPHA = 1.5
DEFAULT_N_QUESTIONS = 20
DEFAULT_KS = "50,100,200,450,750"
DEFAULT_K_MAX = 1000

# Fallback layer when validation_report.json is missing (run validate first).
TRAIT_REGISTRY: dict[str, dict] = {
    "good": {
        "run_id": "dnd_good_scale",
        "layer": 15,
    },
    "evil": {
        "run_id": "dnd_evil",
        "layer": 15,
    },
    "lawful": {
        "run_id": "dnd_lawful",
        "layer": 15,
    },
    "chaotic": {
        "run_id": "dnd_chaotic",
        "layer": 15,
    },
    "male": {
        "run_id": "gender_male",
        "layer": 15,
    },
    "female": {
        "run_id": "gender_female",
        "layer": 15,
    },
}


def sae_id_for_layer(layer: int, width: str = "262k") -> str:
    return f"layer_{layer}_width_{width}_l0_small"


def hidden_state_index(layer: int) -> int:
    return layer + 1


def run_paths(run_id: str, layer: int, *, root: Path | None = None) -> dict[str, Path]:
    base = (root or Path("persona_runs")) / run_id
    sae_dir = base / "sae"
    tag = f"262k_l{layer}"
    return {
        "base": base,
        "bundle": base / "artifacts" / "trait_bundle.json",
        "vectors": base / "vectors" / "persona_vectors.pt",
        "sae_dir": sae_dir,
        "decomp": sae_dir / f"omp_decomposition_{tag}.json",
        "steer": sae_dir / f"omp_steer_results_{tag}.json",
        "geometry": sae_dir / f"omp_geometry_16k_vs_262k_{tag}.json",
        "manifest": sae_dir / "trait_sae_manifest.json",
    }


def load_validate_config(run_id: str, *, root: Path | None = None) -> dict:
    """Load recommended layer, alpha, and n_questions from validate Gate 3 output."""
    report_path = (root or Path("persona_runs")) / run_id / "eval" / "validation_report.json"
    if not report_path.is_file():
        return {}
    data = json.loads(report_path.read_text(encoding="utf-8"))
    out: dict = {}
    if data.get("recommended_layer") is not None:
        out["layer"] = int(data["recommended_layer"])
    if data.get("recommended_alpha") is not None:
        out["alpha"] = float(data["recommended_alpha"])
    if data.get("n_questions") is not None:
        out["n_questions"] = int(data["n_questions"])
    else:
        for gate in data.get("gates") or []:
            if gate.get("gate") == "steering_effectiveness":
                nq = (gate.get("details") or {}).get("n_questions_used")
                if nq is not None:
                    out["n_questions"] = int(nq)
                    break
    out["validation_report"] = report_path
    return out


def print_config_banner(
    cfg: dict,
    *,
    script: str | None = None,
    n_questions: int | None = None,
    n_questions_source: str | None = None,
) -> None:
    """Print a visible config banner so logs show validated vs overridden parameters."""
    nq = n_questions if n_questions is not None else cfg.get("n_questions")
    nq_src = n_questions_source or cfg.get("n_questions_source", "?")
    title = script or "eval"
    lines = [
        f"trait={cfg.get('trait')}  layer={cfg.get('layer')} ({cfg.get('layer_source')})  "
        f"alpha={cfg.get('alpha'):.1f} ({cfg.get('alpha_source')})",
        f"sae={cfg.get('sae_id')}  hs_index={cfg.get('hs_index')}  "
        f"n_questions={nq} ({nq_src})",
    ]
    width = max(len(title), *(len(line) for line in lines)) + 4
    bar = "═" * width
    print(f"╔{bar}╗", flush=True)
    print(f"║ {title:<{width - 2}} ║", flush=True)
    for line in lines:
        print(f"║ {line:<{width - 2}} ║", flush=True)
    print(f"╚{bar}╝", flush=True)


def check_override(
    cfg: dict,
    *,
    cli_layer: int | None = None,
    cli_alpha: float | None = None,
    cli_nq: int | None = None,
) -> None:
    """Warn loudly when CLI args override validated parameters."""
    checks = [
        ("layer", cli_layer, "layer"),
        ("alpha", cli_alpha, "alpha"),
        ("n_questions", cli_nq, "n_questions"),
    ]
    for name, cli_val, cfg_key in checks:
        if cli_val is None or cfg_key not in cfg:
            continue
        cfg_val = cfg[cfg_key]
        if isinstance(cfg_val, float):
            same = abs(float(cli_val) - cfg_val) < 1e-6
        else:
            same = cli_val == cfg_val
        if not same:
            logger.warning(
                "OVERRIDE: CLI --%s=%s differs from %s value %s (%s)",
                name,
                cli_val,
                cfg.get(f"{cfg_key}_source", "?"),
                cfg_val,
                cfg_key,
            )


def resolve_trait(trait: str, *, root: Path | None = None) -> dict:
    key = trait.lower().strip()
    if key not in TRAIT_REGISTRY:
        raise KeyError(f"Unknown trait {trait!r}; choose from {list(TRAIT_REGISTRY)}")
    cfg = TRAIT_REGISTRY[key].copy()
    cfg["trait"] = key
    fallback_layer = int(cfg["layer"])

    vcfg = load_validate_config(cfg["run_id"], root=root)
    if vcfg.get("layer") is not None:
        cfg["layer"] = int(vcfg["layer"])
        cfg["layer_source"] = "validate"
    else:
        cfg["layer"] = fallback_layer
        cfg["layer_source"] = "registry_fallback"
        logger.warning(
            "No validate layer for %s (%s); using TRAIT_REGISTRY layer=%d",
            key,
            cfg["run_id"],
            fallback_layer,
        )

    if vcfg.get("alpha") is not None:
        cfg["alpha"] = float(vcfg["alpha"])
        cfg["alpha_source"] = "validate"
    else:
        cfg["alpha"] = DEFAULT_ALPHA
        cfg["alpha_source"] = "default_fallback"
        logger.warning(
            "No validate alpha for %s (%s); using DEFAULT_ALPHA=%.1f",
            key,
            cfg["run_id"],
            DEFAULT_ALPHA,
        )

    if vcfg.get("n_questions") is not None:
        cfg["n_questions"] = int(vcfg["n_questions"])
        cfg["n_questions_source"] = "validate"
    else:
        cfg["n_questions"] = DEFAULT_N_QUESTIONS
        cfg["n_questions_source"] = "default_fallback"

    cfg["sae_id"] = sae_id_for_layer(cfg["layer"])
    cfg["hs_index"] = hidden_state_index(cfg["layer"])
    paths = run_paths(cfg["run_id"], cfg["layer"], root=root)
    cfg.update(paths)

    logger.info(
        "resolve_trait %s: layer=%d (%s) alpha=%.1f (%s) n_questions=%d (%s)",
        key,
        cfg["layer"],
        cfg["layer_source"],
        cfg["alpha"],
        cfg["alpha_source"],
        cfg["n_questions"],
        cfg["n_questions_source"],
    )
    print_config_banner(cfg)
    return cfg
