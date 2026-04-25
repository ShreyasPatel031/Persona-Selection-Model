import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import login
from pydantic import BaseModel, Field, field_validator
from transformers import TextIteratorStreamer

# ``pipeline`` is imported inside ``lifespan`` so ``import app.main`` does not load
# ``transformers.pipelines`` (heavy; can fail when torchvision/torch versions mismatch).

from app import phase2
from app.persona.response_style import with_paragraph_cap
from app.persona.schemas import PersonaTraitArtifact
MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-3-4b-it")
MAX_NEW_TOKENS = int(os.environ.get("GEMMA_MAX_NEW_TOKENS", "256"))
GATE_AGENT_MAX_TOOL_ROUNDS = int(os.environ.get("GATE_AGENT_MAX_TOOL_ROUNDS", "12"))

STATIC_DIR = Path(__file__).resolve().parent / "static"

_pipe = None
logger = logging.getLogger(__name__)

_persona_steer_lock = Lock()
_persona_v_full_cpu: torch.Tensor | None = None
_persona_bundle_neg_default: str | None = None
_persona_bundle_tried = False

_dnd_play_lock = Lock()
_dnd_play_bundle: dict | None = None


def _default_steer_layer() -> int:
    return int(os.environ.get("PERSONA_STEER_LAYER", "29"))


def _ensure_persona_bundle_neg_default() -> None:
    """Load neg system prompt from PERSONA_STEER_BUNDLE once (for steered chat default)."""
    global _persona_bundle_neg_default, _persona_bundle_tried
    with _persona_steer_lock:
        if _persona_bundle_tried:
            return
        _persona_bundle_tried = True
        bp = os.environ.get("PERSONA_STEER_BUNDLE", "").strip()
        if not bp:
            return
        path = Path(bp).expanduser().resolve()
        if not path.is_file():
            logger.warning("PERSONA_STEER_BUNDLE not found: %s", path)
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            art = PersonaTraitArtifact.model_validate(data)
            _persona_bundle_neg_default = with_paragraph_cap(art.neg_system_prompt)
            logger.info("Persona steer: using neg system default from %s", path)
        except Exception as e:
            logger.warning("PERSONA_STEER_BUNDLE parse failed: %s", e)


def _require_persona_v_cpu() -> torch.Tensor:
    global _persona_v_full_cpu
    with _persona_steer_lock:
        if _persona_v_full_cpu is not None:
            return _persona_v_full_cpu
        pv = os.environ.get("PERSONA_STEER_VECTORS", "").strip()
        if not pv:
            raise HTTPException(
                status_code=503,
                detail="Persona steering not configured: set PERSONA_STEER_VECTORS to persona_vectors.pt "
                "(and optionally PERSONA_STEER_BUNDLE for the contrast neg system prompt).",
            )
        path = Path(pv).expanduser().resolve()
        if not path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"PERSONA_STEER_VECTORS file not found: {path}",
            )
        ckpt = torch.load(path, map_location="cpu")
        if "v" not in ckpt:
            raise HTTPException(
                status_code=500,
                detail="persona_vectors.pt missing key 'v'",
            )
        _persona_v_full_cpu = ckpt["v"].float()
        logger.info("Loaded persona vectors from %s shape=%s", path, tuple(_persona_v_full_cpu.shape))
        return _persona_v_full_cpu


def _pick_pipeline_device() -> int:
    """Use lowest CUDA device (0) when available unless GEMMA_FORCE_CPU is set."""
    if os.environ.get("GEMMA_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return -1
    if torch.cuda.is_available():
        return 0
    return -1


def _hf_login() -> None:
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipe
    _pipe = None
    if os.environ.get("HF_TOKEN"):
        _hf_login()
        dev_id = _pick_pipeline_device()
        if dev_id >= 0:
            dt = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            logger.info("Loading model %s on CUDA:%s (%s)…", MODEL_ID, dev_id, dt)
        else:
            dt = torch.float32
            logger.info("Loading model %s (CPU)…", MODEL_ID)
        from transformers import pipeline

        _pipe = pipeline(
            task="text-generation",
            model=MODEL_ID,
            device=dev_id,
            torch_dtype=dt,
            model_kwargs={"low_cpu_mem_usage": True},
        )
        logger.info("Model loaded.")
        dev = next(_pipe.model.parameters()).device
        phase2.try_load_sae(dev)
    else:
        logger.warning(
            "HF_TOKEN not set; model not loaded. Export HF_TOKEN and restart Uvicorn to enable /chat."
        )
    yield
    _pipe = None


app = FastAPI(lifespan=lifespan)
# Lets static HTML use ?api=http://vm:port when the page is not same-origin (e.g. dev server + VM API).
_cors = os.environ.get("CORS_ALLOW_ORIGINS", "*").strip()
if _cors and _cors != "-":
    _origins = (
        ["*"]
        if _cors == "*"
        else [o.strip() for o in _cors.split(",") if o.strip()]
    )
    if _origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    system: str = Field(
        default="You are a helpful assistant.",
        min_length=1,
        max_length=8000,
    )
    do_sample: bool = False
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Used when do_sample=True (default 1.0 if omitted).",
    )
    seed: int | None = Field(
        default=None,
        description="Optional RNG seed for reproducible sampling.",
    )


class ChatResponse(BaseModel):
    reply: str


_GATE_CHAT_NO_MODEL_MSG = (
    "Model not loaded on this API host (set HF_TOKEN where Uvicorn runs and restart)."
)


class ChatGateChatResponse(BaseModel):
    reply: str
    gates: list[bool]
    trait_alphas: dict[str, float] = Field(
        default_factory=dict,
        description="Per-trait α from dnd_grid_results Pareto corners (internal calibration).",
    )
    planner_raw: str | None = Field(
        default=None,
        description="Raw planner output when gates were model-chosen.",
    )
    error: str | None = Field(
        default=None,
        description="If set, no assistant reply was produced (same conditions as SSE error on /gate-chat/stream).",
    )


class ChatPersonaSteerRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    system: str | None = Field(
        default=None,
        min_length=1,
        max_length=8000,
        description="Omit to use neg prompt from PERSONA_STEER_BUNDLE when set, else helpful default.",
    )
    steer_alpha: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Additive steering strength α at steer_layer (0 = same as plain /chat/stream).",
    )
    steer_layer: int | None = Field(
        default=None,
        ge=0,
        le=127,
        description="Decoder layer; default PERSONA_STEER_LAYER or 29.",
    )
    do_sample: bool = False
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Used when do_sample=True (default 1.0 if omitted).",
    )
    seed: int | None = None


class ChatDndPlaygroundRequest(BaseModel):
    """3×3 D&D alignment cell (LG … CE); streams steered Gemma tokens."""

    alignment: str = Field(
        ...,
        min_length=1,
        max_length=8,
        description="One of LG, NG, CG, LN, N, CN, LE, NE, CE.",
    )
    message: str = Field(..., min_length=1, max_length=8000)
    system: str | None = Field(
        default=None,
        min_length=1,
        max_length=8000,
        description="Omit to use system string from dnd_grid_results.json.",
    )
    do_sample: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    seed: int | None = None


class GateChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1, max_length=32000)


class ChatGateChatRequest(BaseModel):
    """Multi-turn gate self-awareness chat: one agent toggles gates via ``[toggle_gate(...)]``; state persists client-side."""

    messages: list[GateChatMessage] = Field(..., min_length=1)
    gate_state: list[bool] | None = Field(
        default=None,
        description="Gate 1–4 on/off from the previous turn (at most one true; server enforces exclusivity). Omit for all false.",
    )
    system: str | None = Field(
        default=None,
        min_length=1,
        max_length=8000,
        description="Override system when the first message is not system, or replace default system.",
    )
    do_sample: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    seed: int | None = None
    max_new_tokens: int | None = Field(
        default=None,
        ge=8,
        le=2048,
        description="Per internal generation cap (default GEMMA_MAX_NEW_TOKENS). Lower = faster replies.",
    )

    @field_validator("gate_state", mode="before")
    @classmethod
    def _coerce_gate_state(cls, v: Any) -> list[bool] | None:
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("gate_state must be a list of exactly 4 booleans")
        if len(v) != 4:
            raise ValueError("gate_state must be a list of exactly 4 booleans")
        out: list[bool] = []
        for i, x in enumerate(v):
            if isinstance(x, bool):
                out.append(x)
            elif isinstance(x, (int, float)) and int(x) in (0, 1):
                out.append(bool(int(x)))
            elif isinstance(x, str):
                s = x.strip().lower()
                if s in ("true", "1", "yes", "on"):
                    out.append(True)
                elif s in ("false", "0", "no", "off", ""):
                    out.append(False)
                else:
                    raise ValueError(f"gate_state[{i}] is not a boolean")
            else:
                raise ValueError(f"gate_state[{i}] is not a boolean")
        return out


def _normalize_gate_chat_messages(
    body: ChatGateChatRequest,
    default_system: str,
) -> list[dict[str, str]]:
    from app.persona.gate_planner import build_gate_agent_system

    msgs = [{"role": m.role, "content": m.content} for m in body.messages]
    if msgs[0]["role"] != "system":
        sys_text = body.system if body.system is not None else default_system
        msgs.insert(0, {"role": "system", "content": build_gate_agent_system(sys_text)})
    elif body.system is not None:
        msgs[0] = {"role": "system", "content": build_gate_agent_system(body.system)}
    else:
        msgs[0] = {"role": "system", "content": build_gate_agent_system(msgs[0]["content"])}
    if msgs[-1]["role"] != "user":
        raise ValueError(
            "Last message must be from the user (the assistant reply is generated server-side).",
        )
    return msgs


def _initial_gate_state(body: ChatGateChatRequest) -> list[bool]:
    from app.persona.gate_planner import normalize_exclusive_gates

    if body.gate_state is not None:
        return normalize_exclusive_gates(list(body.gate_state))
    return [False, False, False, False]


def _gate_chat_max_new_tokens(body: ChatGateChatRequest) -> int:
    if body.max_new_tokens is not None:
        return int(body.max_new_tokens)
    return MAX_NEW_TOKENS


def _resolve_steered_system(body: ChatPersonaSteerRequest) -> str:
    if body.system is not None:
        return body.system
    _ensure_persona_bundle_neg_default()
    if _persona_bundle_neg_default:
        return _persona_bundle_neg_default
    return "You are a helpful assistant."


class SaeSnapshotRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    system: str = Field(
        default="You are a helpful assistant.",
        min_length=1,
        max_length=8000,
    )
    topk: int = Field(default=24, ge=4, le=128)


class SaeCompareRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    system_a: str = Field(..., min_length=1, max_length=8000)
    system_b: str = Field(..., min_length=1, max_length=8000)
    topk: int = Field(default=24, ge=4, le=128)


def _build_messages(user_text: str, system: str = "You are a helpful assistant.") -> list:
    return [
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            },
        ]
    ]


def _conversation_for_template(
    user_text: str, system: str = "You are a helpful assistant."
) -> list:
    """Single chat (batch item 0) with string contents for apply_chat_template."""
    conv = []
    for m in _build_messages(user_text, system)[0]:
        parts = []
        for block in m.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        conv.append({"role": m["role"], "content": "".join(parts)})
    return conv


def _stream_events(user_text: str, system: str = "You are a helpful assistant."):
    try:
        tokenizer = _pipe.tokenizer
        model = _pipe.model
        conv = _conversation_for_template(user_text, system)
        inputs = tokenizer.apply_chat_template(
            conv,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    gen_kwargs = {
        **inputs,
        "max_new_tokens": MAX_NEW_TOKENS,
        "streamer": streamer,
        "do_sample": False,
        "pad_token_id": pad_id,
    }

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    try:
        for text in streamer:
            if text:
                yield f"data: {json.dumps({'chunk': text})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        thread.join(timeout=7200)
    yield f"data: {json.dumps({'done': True})}\n\n"


_STREAM_END = object()


def _next_stream_chunk(it):
    try:
        return next(it)
    except StopIteration:
        return _STREAM_END


async def _stream_events_async(
    user_text: str, system: str = "You are a helpful assistant."
):
    """Pull sync generator in a thread per chunk so Uvicorn can flush SSE before more tokens."""
    it = _stream_events(user_text, system)
    while True:
        chunk = await asyncio.to_thread(_next_stream_chunk, it)
        if chunk is _STREAM_END:
            break
        yield chunk
        await asyncio.sleep(0)


def _stream_events_persona_steer(body: ChatPersonaSteerRequest):
    """SSE chunks; α=0 delegates to unsteered stream (no persona .pt required)."""
    system = _resolve_steered_system(body)
    alpha = float(body.steer_alpha)
    if alpha == 0.0:
        yield from _stream_events(body.message, system)
        return

    tokenizer = _pipe.tokenizer
    model = _pipe.model
    v_full = _require_persona_v_cpu()
    layer = body.steer_layer if body.steer_layer is not None else _default_steer_layer()
    n_layers = int(v_full.shape[0])
    if layer < 0 or layer >= n_layers:
        yield f"data: {json.dumps({'error': f'steer_layer {layer} out of range [0, {n_layers - 1}]'})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    direction = v_full[layer].to(device=device, dtype=dtype).view(1, 1, -1)

    try:
        conv = _conversation_for_template(body.message, system)
        inputs = tokenizer.apply_chat_template(
            conv,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    from app.persona.steering_demo import _language_model_layers, _steering_hook_fn

    hook_calls = [0]
    hook = _steering_hook_fn(
        alpha,
        direction,
        steer_last_token_only=False,
        hook_calls=hook_calls,
    )
    layers = _language_model_layers(model)
    gen_kwargs = {
        **inputs,
        "max_new_tokens": MAX_NEW_TOKENS,
        "streamer": streamer,
        "do_sample": body.do_sample,
        "pad_token_id": pad_id,
    }
    if body.do_sample:
        gen_kwargs["temperature"] = (
            float(body.temperature) if body.temperature is not None else 1.0
        )

    def run_generate():
        if body.seed is not None:
            torch.manual_seed(int(body.seed))
        model.generate(**gen_kwargs)

    handle = layers[layer].register_forward_hook(hook)
    thread = Thread(target=run_generate)
    thread.start()
    try:
        for text in streamer:
            if text:
                yield f"data: {json.dumps({'chunk': text})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        thread.join(timeout=7200)
        handle.remove()
    if hook_calls[0] == 0:
        yield f"data: {json.dumps({'error': 'Steering hook never ran; check steer_layer vs model.'})}\n\n"
    yield f"data: {json.dumps({'done': True})}\n\n"


async def _stream_events_persona_steer_async(body: ChatPersonaSteerRequest):
    it = _stream_events_persona_steer(body)
    while True:
        chunk = await asyncio.to_thread(_next_stream_chunk, it)
        if chunk is _STREAM_END:
            break
        yield chunk
        await asyncio.sleep(0)


def _ensure_dnd_playground() -> dict:
    """Load dnd_config + grid JSON + four trait vectors (CPU) once."""
    global _dnd_play_bundle
    with _dnd_play_lock:
        if _dnd_play_bundle is not None:
            return _dnd_play_bundle
        from app.persona.dnd_playground import load_playground_bundle

        bundle = load_playground_bundle()
        v_cpu: dict[str, torch.Tensor] = {}
        for name, spec in bundle["traits_cfg"].items():
            p = Path(spec["vectors"]).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"DND vectors missing for {name}: {p}")
            ck = torch.load(p, map_location="cpu", weights_only=False)
            v_cpu[name] = ck["v"].float()
        bundle["v_cpu"] = v_cpu
        _dnd_play_bundle = bundle
        logger.info(
            "D&D playground loaded config=%s grid=%s",
            bundle["config_path"],
            bundle["grid_path"],
        )
        return _dnd_play_bundle


def _stream_events_dnd_playground(body: ChatDndPlaygroundRequest):
    from app.persona.grid_nine import generate_steered_two_axes_stream
    from app.persona.vector_compose import build_positive_direction

    try:
        bundle = _ensure_dnd_playground()
    except FileNotFoundError as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    key = body.alignment.strip().upper()
    presets = bundle["presets"]
    if key not in presets:
        yield f"data: {json.dumps({'error': f'Unknown alignment {key!r}; use one of {sorted(presets)}'})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    if _pipe is None:
        yield f"data: {json.dumps({'error': 'Model not loaded on this API host (set HF_TOKEN where Uvicorn runs and restart).'})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    preset = presets[key]
    trait_a: str = preset["trait_a"]
    trait_b: str = preset["trait_b"]
    traits_cfg = bundle["traits_cfg"]
    if trait_a not in traits_cfg or trait_b not in traits_cfg:
        yield f"data: {json.dumps({'error': f'Missing trait in config: {trait_a}, {trait_b}'})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    spec_a = traits_cfg[trait_a]
    spec_b = traits_cfg[trait_b]
    la = int(spec_a["layer"])
    lb = int(spec_b["layer"])
    v_a = bundle["v_cpu"][trait_a]
    v_b = bundle["v_cpu"][trait_b]

    from app.persona.steering_demo import _language_model_layers

    model = _pipe.model
    tokenizer = _pipe.tokenizer
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    layers = _language_model_layers(model)
    n_layers = len(layers)
    if not (0 <= la < n_layers) or not (0 <= lb < n_layers):
        yield f"data: {json.dumps({'error': f'Layer out of range: {la}, {lb} vs n_layers={n_layers}'})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    d_a = build_positive_direction(v_a, la, device, dtype)
    d_b = build_positive_direction(v_b, lb, device, dtype)
    system = body.system if body.system is not None else bundle["system"]
    alpha_a = float(preset["alpha_a"])
    alpha_b = float(preset["alpha_b"])

    try:
        if body.seed is not None:
            torch.manual_seed(int(body.seed))
        stream_it = generate_steered_two_axes_stream(
            model,
            tokenizer,
            device,
            system,
            body.message,
            layers=layers,
            layer_syc=la,
            direction_syc=d_a,
            alpha_syc=alpha_a,
            layer_chaos=lb,
            direction_chaos=d_b,
            alpha_chaos=alpha_b,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=body.do_sample,
            temperature=float(body.temperature) if body.temperature is not None else 1.0,
        )
        for text in stream_it:
            if text:
                yield f"data: {json.dumps({'chunk': text})}\n\n"
    except Exception as e:
        logger.exception("D&D playground stream failed")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    yield f"data: {json.dumps({'done': True, 'alignment': key, 'alpha_a': alpha_a, 'alpha_b': alpha_b, 'trait_a': trait_a, 'trait_b': trait_b})}\n\n"


async def _stream_events_dnd_playground_async(body: ChatDndPlaygroundRequest):
    it = _stream_events_dnd_playground(body)
    while True:
        chunk = await asyncio.to_thread(_next_stream_chunk, it)
        if chunk is _STREAM_END:
            break
        yield chunk
        await asyncio.sleep(0)


def _stream_events_gate_chat(body: ChatGateChatRequest):
    from app.persona.lm_layers import language_model_layers

    try:
        bundle = _ensure_dnd_playground()
    except FileNotFoundError as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    if _pipe is None:
        yield f"data: {json.dumps({'error': _GATE_CHAT_NO_MODEL_MSG})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    try:
        messages = _normalize_gate_chat_messages(body, bundle["system"])
    except ValueError as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    model = _pipe.model
    tokenizer = _pipe.tokenizer
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    layers = language_model_layers(model)

    trait_alphas: dict[str, float] = dict(bundle.get("trait_alphas") or {})
    gates_in = _initial_gate_state(body)
    gates_out: list[bool] = list(gates_in)

    yield f"data: {json.dumps({'gate_state_in': gates_in, 'trait_alphas': trait_alphas})}\n\n"

    if body.seed is not None:
        torch.manual_seed(int(body.seed))

    temp = float(body.temperature) if body.temperature is not None else 1.0
    cap = _gate_chat_max_new_tokens(body)

    try:
        from app.persona.gate_planner import run_gate_agent_turn_token_level

        working = [dict(m) for m in messages]
        gates_out = list(gates_in)
        for ev in run_gate_agent_turn_token_level(
            messages=working,
            gates=gates_in,
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            layers=layers,
            bundle=bundle,
            trait_alphas=trait_alphas,
            max_new_tokens=cap,
            do_sample=body.do_sample,
            temperature=temp,
        ):
            if ev.get("banner"):
                yield f"data: {json.dumps({'banner': ev['banner']})}\n\n"
            if ev.get("gate_update"):
                yield f"data: {json.dumps({'gate_update': ev['gate_update']})}\n\n"
            chunk = ev.get("chunk")
            if chunk:
                payload: dict[str, Any] = {"chunk": chunk}
                if "segment" in ev:
                    payload["segment"] = ev["segment"]
                yield f"data: {json.dumps(payload)}\n\n"
            if ev.get("final"):
                gates_out = [bool(x) for x in ev["gates"]]
    except Exception as e:
        logger.exception("gate-chat stream failed")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    gates_out = [bool(x) for x in gates_out]
    yield f"data: {json.dumps({'done': True, 'gates': gates_out, 'trait_alphas': trait_alphas})}\n\n"


def _assistant_text(block: dict) -> str:
    """Gemma 3 pipelines may use string content or multimodal [{type,text}, …]."""
    content = block.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(str(c.get("text", "")))
        return "\n".join(parts).strip()
    return ""


def _unwrap_batch_item(item):
    """Pipeline may return [[{generated_text: ...}]] for batch-of-conversations."""
    while isinstance(item, list) and len(item) == 1:
        item = item[0]
    return item


def _extract_reply(raw) -> str:
    if not raw:
        return ""
    first = _unwrap_batch_item(raw[0])
    if isinstance(first, dict) and "generated_text" in first:
        gen = first["generated_text"]
        if isinstance(gen, str):
            return gen.strip()
        if isinstance(gen, list):
            # Flatten one level if the pipeline wrapped messages in an extra list
            blocks = gen
            if len(gen) == 1 and isinstance(gen[0], list):
                blocks = gen[0]
            parts = []
            for block in blocks:
                if isinstance(block, dict) and block.get("role") == "assistant":
                    t = _assistant_text(block)
                    if t:
                        parts.append(t)
            if parts:
                return "\n".join(parts).strip()
        return str(gen).strip()
    return str(first).strip()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/phase2.html")
def phase2_page():
    return FileResponse(STATIC_DIR / "phase2.html")


@app.get("/dnd_playground.html")
def dnd_playground_page():
    return FileResponse(STATIC_DIR / "dnd_playground.html")


@app.get("/dnd/playground/meta")
def dnd_playground_meta():
    """Presets and default question (no model vectors loaded)."""
    try:
        from app.persona.dnd_playground import load_playground_bundle

        b = load_playground_bundle()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e
    return {
        "default_question": b["default_question"],
        "questions_by_alignment": b.get("questions_by_alignment") or {},
        "system": b["system"],
        "alignments": b["presets"],
        "config_path": b["config_path"],
        "grid_path": b["grid_path"],
    }


@app.post("/dnd/playground/stream")
async def dnd_playground_stream(body: ChatDndPlaygroundRequest):
    # Same pattern as gate-chat: always SSE; missing model is an event inside the stream (see _stream_events_dnd_playground).
    return StreamingResponse(
        _stream_events_dnd_playground_async(body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/gate_chat.html")
def gate_chat_page():
    return FileResponse(
        STATIC_DIR / "gate_chat.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/gate-chat/meta")
def gate_chat_meta():
    """Default system prompt and copy for the four-gate chat UI (no vector semantics)."""
    try:
        from app.persona.dnd_playground import load_playground_bundle
        from app.persona.gate_planner import build_gate_agent_system

        b = load_playground_bundle()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    ta = b.get("trait_alphas") or {}
    base = b["system"]
    return {
        "system": base,
        "system_sent_to_model": build_gate_agent_system(base),
        "default_question": b.get("default_question") or "",
        "config_path": b["config_path"],
        "grid_path": b["grid_path"],
        "trait_alphas": ta,
        "intro": (
            "Gates G1–G4: one ON at a time via [toggle_gate(gate=N, state=true/false)]; state persists (you send gate_state). "
            "Active gate uses that trait’s α from the D&D grid corners."
        ),
        "steering_explanation": (
            "Gates are anonymous to the model. Server adds steering vectors; α per trait from dnd_grid_results.json Pareto corners (one gate on = that α; all off = none)."
        ),
    }


@app.post("/gate-chat", response_model=ChatGateChatResponse)
def gate_chat(body: ChatGateChatRequest):
    from app.persona.gate_planner import run_gate_agent_turn
    from app.persona.lm_layers import language_model_layers

    try:
        bundle = _ensure_dnd_playground()
    except FileNotFoundError as e:
        return ChatGateChatResponse(
            reply="",
            gates=[False, False, False, False],
            trait_alphas={},
            planner_raw=None,
            error=str(e),
        )
    try:
        messages = _normalize_gate_chat_messages(body, bundle["system"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    trait_alphas: dict[str, float] = dict(bundle.get("trait_alphas") or {})

    if _pipe is None:
        return ChatGateChatResponse(
            reply="",
            gates=[False, False, False, False],
            trait_alphas=trait_alphas,
            planner_raw=None,
            error=_GATE_CHAT_NO_MODEL_MSG,
        )

    model = _pipe.model
    tokenizer = _pipe.tokenizer
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    layers = language_model_layers(model)

    if body.seed is not None:
        torch.manual_seed(int(body.seed))

    gates_in = _initial_gate_state(body)
    temp = float(body.temperature) if body.temperature is not None else 1.0
    cap = _gate_chat_max_new_tokens(body)
    try:
        working = [dict(m) for m in messages]
        reply, gates_out = run_gate_agent_turn(
            messages=working,
            gates=gates_in,
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            layers=layers,
            bundle=bundle,
            trait_alphas=trait_alphas,
            max_new_tokens=cap,
            do_sample=body.do_sample,
            temperature=temp,
            max_tool_rounds=GATE_AGENT_MAX_TOOL_ROUNDS,
        )
    except Exception as e:
        logger.exception("gate-chat failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    gates_out = [bool(x) for x in gates_out]
    return ChatGateChatResponse(
        reply=reply,
        gates=gates_out,
        trait_alphas=trait_alphas,
        planner_raw=None,
        error=None,
    )


@app.post("/gate-chat/stream")
def gate_chat_stream(body: ChatGateChatRequest):
    # Sync generator + Starlette iterate_in_threadpool: one SSE chunk per yield (token/banner/trace).
    return StreamingResponse(
        _stream_events_gate_chat(body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class RotatingGateTestRequest(BaseModel):
    """Rotating-gate diagnostic: cycles G1→G2→G3→G4 per token."""
    question: str = Field(
        default="Your landlord just raised your rent by an amount that feels unfair, but it's technically within the lease terms you signed.",
        min_length=1,
        max_length=8000,
    )
    system: str | None = Field(default=None, min_length=1, max_length=8000)
    do_sample: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    seed: int | None = None
    max_new_tokens: int = Field(default=128, ge=8, le=2048)


def _stream_rotating_gate_test(body: RotatingGateTestRequest):
    from app.persona.gate_planner import run_rotating_gate_test
    from app.persona.lm_layers import language_model_layers

    try:
        bundle = _ensure_dnd_playground()
    except FileNotFoundError as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    if _pipe is None:
        yield f"data: {json.dumps({'error': _GATE_CHAT_NO_MODEL_MSG})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    model = _pipe.model
    tokenizer = _pipe.tokenizer
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    layers = language_model_layers(model)
    trait_alphas: dict[str, float] = dict(bundle.get("trait_alphas") or {})

    sys_text = body.system if body.system is not None else bundle["system"]
    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": body.question},
    ]

    if body.seed is not None:
        torch.manual_seed(int(body.seed))
    temp = float(body.temperature) if body.temperature is not None else 1.0

    try:
        for ev in run_rotating_gate_test(
            messages=messages,
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            layers=layers,
            bundle=bundle,
            trait_alphas=trait_alphas,
            max_new_tokens=body.max_new_tokens,
            do_sample=body.do_sample,
            temperature=temp,
        ):
            if ev.get("banner"):
                yield f"data: {json.dumps({'banner': ev['banner']})}\n\n"
            chunk = ev.get("chunk")
            if chunk:
                payload: dict[str, Any] = {"chunk": chunk}
                if "gate_for_token" in ev:
                    payload["gate_for_token"] = ev["gate_for_token"]
                yield f"data: {json.dumps(payload)}\n\n"
            if ev.get("final"):
                yield f"data: {json.dumps({'done': True, 'reply': ev.get('reply', ''), 'token_gate_log': ev.get('token_gate_log', []), 'trait_alphas': trait_alphas})}\n\n"
    except Exception as e:
        logger.exception("rotating gate test failed")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"


@app.post("/gate-chat/test-rotating")
def gate_chat_test_rotating(body: RotatingGateTestRequest):
    return StreamingResponse(
        _stream_rotating_gate_test(body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _pipe is not None,
        "phase2_sae": phase2.sae_status(),
        "persona_steer": {
            "vectors_env_set": bool(os.environ.get("PERSONA_STEER_VECTORS", "").strip()),
            "bundle_env_set": bool(os.environ.get("PERSONA_STEER_BUNDLE", "").strip()),
            "default_layer": _default_steer_layer(),
            "vectors_loaded_in_memory": _persona_v_full_cpu is not None,
        },
        "dnd_playground": {
            "bundle_loaded": _dnd_play_bundle is not None,
        },
        "gate_chat": {
            "dnd_bundle_loaded": _dnd_play_bundle is not None,
        },
    }


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    if _pipe is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. On the VM: export HF_TOKEN=… then restart Uvicorn.",
        )
    try:
        if body.seed is not None:
            torch.manual_seed(int(body.seed))
        gen_kwargs: dict = {
            "max_new_tokens": MAX_NEW_TOKENS,
        }
        if body.do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = (
                float(body.temperature) if body.temperature is not None else 1.0
            )
        else:
            gen_kwargs["do_sample"] = False
        out = _pipe(
            _build_messages(body.message, body.system),
            **gen_kwargs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return ChatResponse(reply=_extract_reply(out))


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    if _pipe is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. On the VM: export HF_TOKEN=… then restart Uvicorn.",
        )
    return StreamingResponse(
        _stream_events_async(body.message, body.system),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat/persona-steer", response_model=ChatResponse)
def chat_persona_steer(body: ChatPersonaSteerRequest):
    if _pipe is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. On the VM: export HF_TOKEN=… then restart Uvicorn.",
        )
    system = _resolve_steered_system(body)
    try:
        if body.steer_alpha == 0.0:
            if body.seed is not None:
                torch.manual_seed(int(body.seed))
            gen_kwargs: dict = {"max_new_tokens": MAX_NEW_TOKENS}
            if body.do_sample:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = (
                    float(body.temperature) if body.temperature is not None else 1.0
                )
            else:
                gen_kwargs["do_sample"] = False
            out = _pipe(
                _build_messages(body.message, system),
                **gen_kwargs,
            )
            return ChatResponse(reply=_extract_reply(out))
        v_full = _require_persona_v_cpu()
        layer = body.steer_layer if body.steer_layer is not None else _default_steer_layer()
        n_layers = int(v_full.shape[0])
        if layer < 0 or layer >= n_layers:
            raise HTTPException(
                status_code=400,
                detail=f"steer_layer {layer} out of range [0, {n_layers - 1}]",
            )
        model = _pipe.model
        tokenizer = _pipe.tokenizer
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        direction = v_full[layer].to(device=device, dtype=dtype).view(1, 1, -1)
        if body.seed is not None:
            torch.manual_seed(int(body.seed))
        from app.persona.quality_gates import _generate_steered

        reply = _generate_steered(
            model,
            tokenizer,
            device,
            system,
            body.message,
            layer_idx=layer,
            direction=direction,
            alpha=float(body.steer_alpha),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=body.do_sample,
            temperature=float(body.temperature) if body.temperature is not None else 1.0,
        )
        return ChatResponse(reply=reply)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/chat/persona-steer/stream")
async def chat_persona_steer_stream(body: ChatPersonaSteerRequest):
    if _pipe is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. On the VM: export HF_TOKEN=… then restart Uvicorn.",
        )
    if body.steer_alpha > 0:
        _require_persona_v_cpu()
    return StreamingResponse(
        _stream_events_persona_steer_async(body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/phase2/sae_snapshot")
def phase2_sae_snapshot(body: SaeSnapshotRequest):
    if _pipe is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )
    if not phase2.sae_loaded():
        raise HTTPException(
            status_code=503,
            detail="SAE not loaded. Install sae-lens, check logs, or set SAE_RELEASE/SAE_ID. "
            "Use DISABLE_SAE=1 to skip loading.",
        )
    try:
        return phase2.compute_snapshot(_pipe, body.system, body.message, body.topk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/phase2/sae_compare")
def phase2_sae_compare(body: SaeCompareRequest):
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not phase2.sae_loaded():
        raise HTTPException(status_code=503, detail="SAE not loaded.")
    try:
        return phase2.compute_compare(
            _pipe,
            body.message,
            body.system_a,
            body.system_b,
            body.topk,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
