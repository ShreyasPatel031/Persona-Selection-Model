#!/usr/bin/env python3
"""OCEAN MDS demo on Colab T4: extract vectors, find working (layer, α), emit 6 replies.

Search order (as requested):
  1) default: layer 15, α ∈ {2,4,8}
  2) expand mid-layers
  3) expand α
  4) debug variants until a clear baseline≠steered effect appears

MDS = (μ(S↑) − μ(S↓)) / 2 on residual stream at each layer (paper-style statement prefills).
Steering uses raw α·v (not unit-normalized).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "unsloth/gemma-3-4b-it")
OUT = Path(os.environ.get("PERSONA_OUT", "/content/ocean_mds_demo.json"))
N_STMT = int(os.environ.get("OCEAN_N_STMT", "40"))  # per pole
MAX_NEW = int(os.environ.get("PERSONA_MAX_NEW", "90"))
PROBE_Q = os.environ.get(
    "OCEAN_PROBE_Q",
    "Write 3-4 sentences about how you spent last Saturday and what mattered to you that day.",
)

# Neutral system so trait shows via residual injection, not prompt persona.
SYSTEM = (
    "You are a helpful assistant. Answer in one short paragraph of plain prose "
    "(3-5 sentences). Do not mention personality traits or scoring."
)

# Compact first-person banks derived from OCEAN facets (IPIP-style).
OCEAN_STATEMENTS: dict[str, dict[str, list[str]]] = {
    "openness": {
        "up": [
            "I love exploring unfamiliar ideas just for the joy of it.",
            "I get excited by abstract theories and unusual art.",
            "I enjoy changing my routines to try something new.",
            "I daydream about alternate possibilities often.",
            "I seek out books and films that challenge my worldview.",
            "I am curious about cultures very different from my own.",
            "I like improvising rather than sticking to a script.",
            "I find beauty in strange and unconventional things.",
            "I question tradition when it blocks creative thought.",
            "I enjoy philosophical conversations that go nowhere practical.",
            "I invent little experiments in my daily life.",
            "I am open to revising my beliefs when evidence shifts.",
            "I get absorbed in imaginative projects for hours.",
            "I prefer novelty over repeating what already works.",
            "I notice subtle patterns others overlook.",
            "I like learning random skills with no career payoff.",
            "I feel alive when brainstorming wild what-ifs.",
            "I appreciate ambiguity instead of needing neat answers.",
            "I rearrange my space to spark fresh perspectives.",
            "I chase intellectual rabbit holes enthusiastically.",
            "I enjoy avant-garde music and experimental design.",
            "I treat travel as a way to rethink who I am.",
            "I write down odd ideas before they disappear.",
            "I connect distant concepts into new metaphors.",
            "I welcome criticism that expands how I see a problem.",
            "I prefer deep originality over polished convention.",
            "I get restless when everything feels too predictable.",
            "I explore hobbies that feel slightly intimidating.",
            "I like art that unsettles me a little.",
            "I invent stories in my head while walking.",
            "I am drawn to speculative and futuristic thinking.",
            "I collect experiences more than possessions.",
            "I ask 'what if' even about ordinary chores.",
            "I enjoy debating ideas without needing to win.",
            "I stretch my comfort zone on purpose.",
            "I find inspiration in unexpected places.",
            "I redesign processes just to see if they can be better.",
            "I value imaginative playfulness in serious work.",
            "I keep learning languages and tools for curiosity.",
            "I feel most myself when inventing something new.",
        ],
        "down": [
            "I prefer familiar routines and proven methods.",
            "I stick to practical facts over abstract speculation.",
            "I dislike changing plans unless I must.",
            "I have little interest in unusual art or theory.",
            "I trust tradition more than experimental ideas.",
            "I want clear answers, not open-ended brainstorming.",
            "I avoid hobbies that feel weird or impractical.",
            "I keep my beliefs stable and rarely revise them.",
            "I focus on what works today, not distant what-ifs.",
            "I find fantasy and speculation a waste of time.",
            "I like repeating successful patterns.",
            "I prefer concrete tasks with known outcomes.",
            "I am uncomfortable with ambiguity.",
            "I choose conventional styles over novelty.",
            "I do not chase intellectual rabbit holes.",
            "I stick to mainstream music and entertainment.",
            "I travel only when it is convenient and familiar.",
            "I rarely invent metaphors or imaginative frames.",
            "I dislike rearranging things just for inspiration.",
            "I want efficiency, not creative exploration.",
            "I avoid avant-garde or unsettling art.",
            "I keep conversations practical and grounded.",
            "I do not enjoy philosophical dead ends.",
            "I prefer polished convention over originality.",
            "I feel calm when everything is predictable.",
            "I skip intimidating new hobbies.",
            "I do not invent stories while walking.",
            "I am uninterested in futuristic speculation.",
            "I collect useful tools, not experiences for curiosity.",
            "I rarely ask 'what if' about chores.",
            "I debate only when a decision is required.",
            "I stay inside my comfort zone.",
            "I get inspiration from manuals, not accidents.",
            "I leave working processes alone.",
            "I dislike playful digressions at work.",
            "I learn only skills with clear payoff.",
            "I feel most myself when things stay steady.",
            "I distrust wild brainstorming sessions.",
            "I prefer checklists to improvisation.",
            "I find abstract theory boring.",
        ],
    },
    "conscientiousness": {
        "up": [
            "I finish what I start, even when it is tedious.",
            "I plan my week carefully and stick to it.",
            "I keep my commitments without reminders.",
            "I organize my workspace so nothing is misplaced.",
            "I double-check details before I submit work.",
            "I set goals and track progress daily.",
            "I resist distractions until the priority task is done.",
            "I arrive early rather than risk being late.",
            "I break big projects into ordered steps.",
            "I feel uneasy leaving loose ends unresolved.",
            "I maintain habits that support long-term aims.",
            "I prepare backup plans for important deadlines.",
            "I keep records so I can audit my own work.",
            "I prefer discipline over impulsive shortcuts.",
            "I clean up messes immediately.",
            "I schedule rest so it does not derail productivity.",
            "I hold myself to high standards of accuracy.",
            "I follow through on small promises as seriously as big ones.",
            "I review checklists before I call something complete.",
            "I budget time with buffers for delays.",
            "I put tools back where they belong.",
            "I prioritize duties over momentary fun.",
            "I update plans when reality changes.",
            "I take responsibility for mistakes and correct them.",
            "I avoid procrastination by starting early.",
            "I measure outcomes against clear criteria.",
            "I keep financial and personal files orderly.",
            "I practice self-control around temptations.",
            "I finish admin tasks the same day when possible.",
            "I prepare materials the night before.",
            "I define 'done' before I begin.",
            "I maintain routines that protect my focus.",
            "I document decisions for later review.",
            "I treat reliability as part of my identity.",
            "I plan contingencies for travel and meetings.",
            "I keep my inbox and tasks triaged.",
            "I prefer methodical progress to chaotic bursts.",
            "I enforce personal deadlines even without a boss.",
            "I inspect quality before celebrating.",
            "I align daily actions with longer goals.",
        ],
        "down": [
            "I often start things and leave them unfinished.",
            "I dislike rigid schedules and detailed plans.",
            "I need reminders to keep commitments.",
            "My workspace stays cluttered and searching wastes time.",
            "I submit work without careful double-checks.",
            "I set goals vaguely and rarely track them.",
            "I get pulled into distractions easily.",
            "I am frequently late or last-minute.",
            "I jump into projects without ordered steps.",
            "I tolerate loose ends for a long time.",
            "I struggle to maintain productive habits.",
            "I rarely prepare backup plans.",
            "I keep few records of what I did.",
            "I take impulsive shortcuts when impatient.",
            "I leave messes for later.",
            "Rest and fun regularly derail my tasks.",
            "I am okay with 'good enough' accuracy.",
            "Small promises slip my mind.",
            "I skip checklists and hope for the best.",
            "I underestimate how long things take.",
            "I leave tools wherever I last used them.",
            "I choose momentary fun over duties.",
            "I resist updating plans even when needed.",
            "I dodge responsibility when mistakes happen.",
            "I procrastinate until pressure forces me.",
            "I avoid clear criteria for done.",
            "My files and finances stay messy.",
            "Temptations usually win.",
            "Admin tasks pile up for weeks.",
            "I scramble the morning of deadlines.",
            "I begin without defining done.",
            "Routines feel oppressive so I abandon them.",
            "I rarely document decisions.",
            "Reliability is not my strong suit.",
            "I wing travel and meetings.",
            "My inbox is chaotic.",
            "I work in chaotic bursts then stall.",
            "Without external pressure I miss deadlines.",
            "I celebrate before inspecting quality.",
            "Daily actions drift from my longer goals.",
        ],
    },
    "extraversion": {
        "up": [
            "I feel energized after spending time with lots of people.",
            "I start conversations with strangers easily.",
            "I like being the center of lively gatherings.",
            "I speak up quickly in group discussions.",
            "I seek parties, crowds, and social events.",
            "I express enthusiasm openly and loudly.",
            "I prefer working collaboratively in busy rooms.",
            "I make new friends without much effort.",
            "I enjoy leading group activities.",
            "I think out loud and invite reactions.",
            "I fill silence with jokes or stories.",
            "I feel restless when I am alone too long.",
            "I greet people warmly and physically expressively.",
            "I volunteer for presentations and performances.",
            "I chase exciting, high-stimulation plans.",
            "I talk through problems with others first.",
            "I thrive on networking and meeting new contacts.",
            "I laugh easily and share that energy.",
            "I prefer open-plan social spaces.",
            "I initiate plans instead of waiting to be invited.",
            "I enjoy debates that stay friendly and animated.",
            "I feel confident introducing myself anywhere.",
            "I recharge by going out, not staying in.",
            "I keep a wide circle of acquaintances.",
            "I show emotion on my face and in my voice.",
            "I like team sports and group hobbies.",
            "I speak with a strong, carrying voice.",
            "I seek attention in playful ways.",
            "I am quick to RSVP yes to social invites.",
            "I feel bored without interpersonal buzz.",
            "I host gatherings often.",
            "I share personal news freely.",
            "I prefer calls and meetups over solo texts.",
            "I jump into conversations midstream.",
            "I enjoy being recognized in a crowd.",
            "I take social risks without overthinking.",
            "I am talkative even with new coworkers.",
            "I plan weekends around people.",
            "I feel alive under bright lights and noise.",
            "I connect quickly and move toward friendship.",
        ],
        "down": [
            "I feel drained after lots of social time.",
            "I rarely start conversations with strangers.",
            "I avoid being the center of gatherings.",
            "I stay quiet in group discussions.",
            "I prefer quiet evenings over parties.",
            "I keep enthusiasm muted and private.",
            "I work best alone in calm spaces.",
            "I make friends slowly, if at all.",
            "I dislike leading group activities.",
            "I think privately before speaking.",
            "I am comfortable with long silences.",
            "I need substantial alone time.",
            "I greet people reservedly.",
            "I avoid presentations when possible.",
            "I prefer low-stimulation plans.",
            "I solve problems alone first.",
            "Networking feels exhausting.",
            "I laugh softly and rarely perform for a room.",
            "I prefer private corners to open social spaces.",
            "I wait to be invited rather than initiating.",
            "I avoid animated debates.",
            "Introducing myself feels effortful.",
            "I recharge by staying in.",
            "I keep a small circle.",
            "I hide emotion behind a calm face.",
            "I prefer solo hobbies.",
            "I speak softly.",
            "I avoid seeking attention.",
            "I often decline social invites.",
            "I feel fine without interpersonal buzz.",
            "I rarely host gatherings.",
            "I keep personal news private.",
            "I prefer texts over calls and meetups.",
            "I do not jump into others' conversations.",
            "I dislike being recognized in a crowd.",
            "I overthink social risks.",
            "I am reserved with new coworkers.",
            "I plan weekends around solitude.",
            "Bright lights and noise wear me out.",
            "I connect slowly and carefully.",
        ],
    },
    "agreeableness": {
        "up": [
            "I assume others have good intentions.",
            "I go out of my way to help even when inconvenient.",
            "I avoid conflict and look for compromise.",
            "I forgive people quickly after disagreements.",
            "I speak kindly even when I disagree.",
            "I put group harmony above winning an argument.",
            "I notice others' feelings and respond gently.",
            "I share credit and downplay my own role.",
            "I trust people until given a reason not to.",
            "I volunteer help without being asked.",
            "I apologize first to repair a rift.",
            "I listen more than I interrupt.",
            "I give people the benefit of the doubt.",
            "I dislike exploiting anyone's weakness.",
            "I prefer cooperation to competition.",
            "I comfort friends when they struggle.",
            "I keep promises that protect others' wellbeing.",
            "I soften criticism with care.",
            "I enjoy making others feel included.",
            "I avoid harsh judgments about character.",
            "I donate time or resources when I can.",
            "I respect boundaries and ask before pushing.",
            "I celebrate others' successes sincerely.",
            "I stay polite under frustration.",
            "I try to see disputes from both sides.",
            "I avoid sarcasm that could hurt someone.",
            "I offer second chances readily.",
            "I prioritize fairness that is warm, not cold.",
            "I help strangers with small favors.",
            "I keep confidences carefully.",
            "I thank people often and specifically.",
            "I adjust plans to accommodate others.",
            "I dislike dominating conversations.",
            "I express sympathy openly.",
            "I seek win-win outcomes.",
            "I refrain from gossip that damages reputations.",
            "I am patient with mistakes.",
            "I treat service workers with extra kindness.",
            "I feel uneasy when someone is left out.",
            "I choose compassion when rules feel cruel.",
        ],
        "down": [
            "I am skeptical of others' motives.",
            "I help only when it benefits me.",
            "I confront conflict bluntly and push to win.",
            "I hold grudges after disagreements.",
            "I speak harshly when I disagree.",
            "I value winning arguments over harmony.",
            "I ignore others' feelings if they slow me down.",
            "I take credit and protect my reputation first.",
            "I distrust people by default.",
            "I wait to be asked and often refuse help.",
            "I rarely apologize first.",
            "I interrupt to make my point.",
            "I assume the worst in ambiguous situations.",
            "I exploit weaknesses if it advances my goals.",
            "I prefer competition to cooperation.",
            "I offer little comfort when friends struggle.",
            "I break soft promises when inconvenient.",
            "I criticize without softening.",
            "I do not work to include outsiders.",
            "I judge character quickly and harshly.",
            "I rarely donate time or resources.",
            "I push past boundaries to get results.",
            "I envy others' successes more than I celebrate them.",
            "I get rude under frustration.",
            "I see disputes only from my side.",
            "I use sarcasm as a weapon.",
            "I rarely offer second chances.",
            "I prefer cold efficiency over warm fairness.",
            "I ignore strangers needing small favors.",
            "I share confidences when useful.",
            "I rarely thank people.",
            "I refuse to adjust plans for others.",
            "I dominate conversations.",
            "I withhold sympathy.",
            "I seek outcomes that favor me.",
            "I gossip when it is advantageous.",
            "I am impatient with mistakes.",
            "I treat service interactions transactionally.",
            "I do not notice who is left out.",
            "I enforce rules even when they feel cruel.",
        ],
    },
    "neuroticism": {
        "up": [
            "I worry about things going wrong even on calm days.",
            "I feel anxious before ordinary meetings.",
            "Small setbacks stay in my mind for hours.",
            "I get irritated more quickly than I want to admit.",
            "I replay conversations looking for mistakes.",
            "I feel tense in my body when plans change.",
            "I fear embarrassment in social settings.",
            "I doubt myself after mild criticism.",
            "I catastrophize unlikely risks.",
            "I have trouble winding down at night.",
            "I feel emotionally raw after conflict.",
            "I check repeatedly that I locked the door.",
            "I take neutral feedback as a personal threat.",
            "I feel overwhelmed by stacked minor tasks.",
            "I cry or shut down more easily under stress.",
            "I need reassurance that I am still okay.",
            "I brace for rejection before it happens.",
            "I get moody without a clear external cause.",
            "I feel unsafe when I lack control.",
            "I ruminate on past errors.",
            "I am highly sensitive to tone of voice.",
            "I avoid situations that might expose me.",
            "I feel a knot in my stomach before travel.",
            "I interpret delays as signs of failure.",
            "I struggle to stay calm during uncertainty.",
            "I get jealous or insecure in relationships easily.",
            "I feel drained by my own worry loops.",
            "I overprepare because I fear looking incompetent.",
            "I notice every sign of possible disapproval.",
            "I have spikes of panic that pass slowly.",
            "I feel fragile after a hard day.",
            "I expect disappointment so it hurts less.",
            "I am restless when waiting for news.",
            "I personalize problems that are not about me.",
            "I find it hard to trust that things will be fine.",
            "I get stuck in what-if fear spirals.",
            "I sleep poorly when something is unresolved.",
            "I feel easily threatened by change.",
            "I need time alone to recover from stress.",
            "I experience emotions more intensely than peers.",
        ],
        "down": [
            "I stay calm even when plans wobble.",
            "Ordinary meetings do not make me anxious.",
            "Small setbacks bounce off quickly.",
            "I rarely get irritated over minor issues.",
            "I do not replay conversations for mistakes.",
            "Plan changes do not tense up my body.",
            "I am not preoccupied with embarrassment.",
            "Mild criticism does not shake my self-view.",
            "I keep unlikely risks in perspective.",
            "I wind down easily at night.",
            "Conflict does not leave me emotionally raw.",
            "I lock the door once and move on.",
            "Neutral feedback stays neutral to me.",
            "Stacked minor tasks do not overwhelm me.",
            "Under stress I stay steady rather than shutting down.",
            "I do not need constant reassurance.",
            "I do not brace for rejection by default.",
            "My mood stays even without a clear cause.",
            "I tolerate lacking control without panic.",
            "I leave past errors in the past.",
            "Tone of voice rarely unsettles me.",
            "I face exposing situations with composure.",
            "Travel does not knot my stomach.",
            "Delays are just delays, not failure.",
            "Uncertainty does not steal my calm.",
            "I am not easily jealous or insecure.",
            "I am not drained by worry loops.",
            "I prepare adequately without fear spirals.",
            "I do not hunt for signs of disapproval.",
            "Panic spikes are rare for me.",
            "Hard days do not leave me fragile.",
            "I do not pre-load disappointment.",
            "Waiting for news does not make me restless.",
            "I do not personalize unrelated problems.",
            "I trust that most things will be fine.",
            "I interrupt what-if fear spirals early.",
            "Unresolved items rarely ruin my sleep.",
            "Change does not feel threatening.",
            "I recover from stress without long isolation.",
            "My emotions stay proportionate to events.",
        ],
    },
}

TRAIT_MARKERS: dict[str, list[str]] = {
    "openness": [
        "curious", "novel", "imagin", "explor", "creativ", "idea", "art", "wonder",
        "possibil", "experiment", "unusual", "inspir",
    ],
    "conscientiousness": [
        "plan", "organiz", "checklist", "responsib", "disciplin", "schedule", "goal",
        "order", "careful", "dut", "prepar", "routine",
    ],
    "extraversion": [
        "friend", "people", "party", "social", "gather", "talk", "energetic", "outgoing",
        "crowd", "meet", "together", "chat",
    ],
    "agreeableness": [
        "kind", "help", "compassion", "care", "gentle", "harmon", "forgiv", "warm",
        "support", "empath", "considerate", "together",
    ],
    "neuroticism": [
        "worr", "anxi", "stress", "fear", "nervous", "doubt", "overwhelm", "tense",
        "insecur", "panic", "uneasy", "rumin",
    ],
}


def language_model_layers(m):
    if hasattr(m, "model") and m.model is not None:
        inner = m.model
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return inner.language_model.layers
        if hasattr(inner, "layers"):
            return inner.layers
    raise RuntimeError("Could not find decoder layers")


def pick_dtype():
    if os.environ.get("PERSONA_CUDA_ALLOW_FP16", "").lower() in ("1", "true", "yes"):
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def make_hook(direction: torch.Tensor, alpha: float):
    def hook(_module, _inp, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h + alpha * direction.to(device=h.device, dtype=h.dtype)
            return (h,) + output[1:]
        return output + alpha * direction.to(device=output.device, dtype=output.dtype)

    return hook


def encode_chat(tokenizer, system: str, user: str, device):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    raw = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if isinstance(raw, torch.Tensor):
        ids = raw.to(device)
    else:
        ids = raw["input_ids"].to(device)
    attn = torch.ones_like(ids)
    return ids, attn


def encode_prefill_statement(tokenizer, statement: str, device):
    """MDS 's' mode: Tell me about yourself. + statement as assistant prefill."""
    messages = [
        {"role": "system", "content": "You are a person."},
        {"role": "user", "content": "Tell me about yourself."},
        {"role": "assistant", "content": statement},
    ]
    raw = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    )
    if isinstance(raw, torch.Tensor):
        ids = raw.to(device)
    else:
        ids = raw["input_ids"].to(device)
    # prompt-only length (system+user+generation prompt) for assistant-span mean
    prompt_msgs = [
        {"role": "system", "content": "You are a person."},
        {"role": "user", "content": "Tell me about yourself."},
    ]
    praw = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if isinstance(praw, torch.Tensor):
        plen = int(praw.shape[-1])
    else:
        plen = int(praw["input_ids"].shape[-1])
    return ids, plen


@torch.inference_mode()
def mean_assistant_residual(model, layers, ids, plen: int) -> torch.Tensor:
    """Return [n_layers, hidden] mean residual over assistant tokens."""
    captured: dict[int, torch.Tensor] = {}

    def make_cap(i):
        def hook(_m, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            # mean over assistant token positions
            if h.shape[1] > plen:
                captured[i] = h[0, plen:].float().mean(dim=0).detach().cpu()
            else:
                captured[i] = h[0, -1].float().detach().cpu()

        return hook

    handles = [layer.register_forward_hook(make_cap(i)) for i, layer in enumerate(layers)]
    try:
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    stacked = torch.stack([captured[i] for i in range(len(layers))], dim=0)
    return stacked


@torch.inference_mode()
def extract_mds_for_trait(model, tokenizer, layers, device, trait: str, n: int) -> torch.Tensor:
    up = OCEAN_STATEMENTS[trait]["up"][:n]
    down = OCEAN_STATEMENTS[trait]["down"][:n]
    acc_up = None
    acc_down = None
    for i, stmt in enumerate(up):
        ids, plen = encode_prefill_statement(tokenizer, stmt, device)
        v = mean_assistant_residual(model, layers, ids, plen)
        acc_up = v if acc_up is None else acc_up + v
        if (i + 1) % 10 == 0:
            print(f"  {trait} up {i+1}/{len(up)}", flush=True)
    for i, stmt in enumerate(down):
        ids, plen = encode_prefill_statement(tokenizer, stmt, device)
        v = mean_assistant_residual(model, layers, ids, plen)
        acc_down = v if acc_down is None else acc_down + v
        if (i + 1) % 10 == 0:
            print(f"  {trait} down {i+1}/{len(down)}", flush=True)
    mu_up = acc_up / len(up)
    mu_down = acc_down / len(down)
    # Paper MDS: (μ↑ − μ↓) / 2
    return (mu_up - mu_down) / 2.0


@torch.inference_mode()
def generate(model, tokenizer, layers, device, question: str, direction=None, alpha=0.0, layer=None):
    ids, attn = encode_chat(tokenizer, SYSTEM, question, device)
    handle = None
    if direction is not None and alpha != 0.0 and layer is not None:
        handle = layers[layer].register_forward_hook(make_hook(direction, alpha))
    try:
        out = model.generate(
            input_ids=ids,
            attention_mask=attn,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        if handle is not None:
            handle.remove()
    new = out[0, ids.shape[-1] :]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


def marker_hits(text: str, trait: str) -> int:
    t = text.lower()
    return sum(1 for m in TRAIT_MARKERS[trait] if m in t)


def effect_score(baseline: str, steered: str, trait: str) -> dict:
    if not steered:
        return {"ok": False, "reason": "empty", "diff_chars": 0, "markers": 0}
    if steered == baseline:
        return {"ok": False, "reason": "identical", "diff_chars": 0, "markers": 0}
    # crude difference size
    diff = abs(len(steered) - len(baseline)) + sum(
        1 for a, b in zip(steered, baseline) if a != b
    )
    # repetition / collapse heuristic
    toks = re.findall(r"[A-Za-z']+", steered.lower())
    uniq = len(set(toks)) / max(len(toks), 1)
    markers = marker_hits(steered, trait)
    base_markers = marker_hits(baseline, trait)
    ok = (
        diff >= 40
        and uniq >= 0.35
        and len(steered.split()) >= 12
        and (markers > base_markers or diff >= 120)
    )
    return {
        "ok": ok,
        "reason": "ok" if ok else "weak",
        "diff_chars": int(diff),
        "markers": markers,
        "base_markers": base_markers,
        "uniq_ratio": round(uniq, 3),
    }


def main() -> int:
    assert torch.cuda.is_available(), "CUDA required"
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token)
        print("HF login OK", flush=True)

    dtype = pick_dtype()
    print("dtype", dtype, "model", MODEL_ID, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=token,
    )
    model.eval()
    layers = language_model_layers(model)
    device = next(model.parameters()).device
    n_layers = len(layers)
    print("layers", n_layers, "alloc_GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)

    traits = list(OCEAN_STATEMENTS.keys())
    vectors: dict[str, torch.Tensor] = {}
    for trait in traits:
        print(f"\n=== extract MDS {trait} n={N_STMT} ===", flush=True)
        v = extract_mds_for_trait(model, tokenizer, layers, device, trait, N_STMT)
        vectors[trait] = v
        print(
            f"  norms L15={float(v[min(15, n_layers-1)].norm()):.2f} "
            f"mid={float(v[n_layers//2].norm()):.2f}",
            flush=True,
        )

    # persist vectors
    vec_path = Path("/content/ocean_mds_vectors.pt")
    torch.save({t: vectors[t] for t in traits}, vec_path)
    print("wrote", vec_path, flush=True)

    print("\n=== baseline ===", flush=True)
    baseline = generate(model, tokenizer, layers, device, PROBE_Q)
    print(baseline, flush=True)

    # Search schedule
    default_layer = 15 if n_layers > 15 else n_layers // 2
    mid_layers = sorted(
        {
            default_layer,
            max(2, n_layers // 3),
            n_layers // 2,
            (2 * n_layers) // 3,
            min(n_layers - 2, (3 * n_layers) // 4),
            min(n_layers - 1, default_layer + 3),
            max(0, default_layer - 3),
        }
    )
    # Keep unique mid-ish
    mid_layers = [L for L in mid_layers if 0 <= L < n_layers]

    phases = [
        ("default_L15", [default_layer], [2.0, 4.0, 8.0]),
        ("expand_layers", mid_layers, [2.0, 4.0, 8.0]),
        ("expand_alpha", mid_layers, [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0]),
    ]

    found: dict[str, dict] = {}
    search_log = []

    for phase_name, layer_list, alphas in phases:
        pending = [t for t in traits if t not in found]
        if not pending:
            break
        print(f"\n===== PHASE {phase_name} pending={pending} =====", flush=True)
        for trait in pending:
            best = None
            for layer in layer_list:
                direction = vectors[trait][layer]
                for alpha in alphas:
                    steered = generate(
                        model, tokenizer, layers, device, PROBE_Q, direction, alpha, layer
                    )
                    score = effect_score(baseline, steered, trait)
                    entry = {
                        "phase": phase_name,
                        "trait": trait,
                        "layer": layer,
                        "alpha": alpha,
                        "score": score,
                        "reply": steered,
                    }
                    search_log.append(
                        {k: entry[k] for k in ("phase", "trait", "layer", "alpha", "score")}
                    )
                    print(
                        f"[{phase_name}] {trait} L{layer} a={alpha} "
                        f"ok={score['ok']} diff={score['diff_chars']} "
                        f"markers={score['markers']}/{score.get('base_markers')}",
                        flush=True,
                    )
                    if score["ok"]:
                        # prefer stronger marker lift then larger diff
                        key = (
                            score["markers"] - score.get("base_markers", 0),
                            score["diff_chars"],
                        )
                        if best is None or key > best[0]:
                            best = (key, entry)
            if best is not None:
                found[trait] = best[1]
                print(
                    f"FOUND {trait} @ L{best[1]['layer']} α={best[1]['alpha']}",
                    flush=True,
                )

    # Debug phase if still missing
    missing = [t for t in traits if t not in found]
    if missing:
        print(f"\n===== PHASE debug missing={missing} =====", flush=True)
        # Try last-token-only style by using larger α and more layers including slightly earlier/later
        debug_layers = sorted(
            set(mid_layers + [max(0, default_layer - 6), min(n_layers - 1, default_layer + 6)])
        )
        debug_alphas = [6.0, 10.0, 20.0, 32.0, 48.0]
        for trait in missing:
            best = None
            for layer in debug_layers:
                direction = vectors[trait][layer]
                # also try 2x raw vector (equivalent to dropping the /2 in MDS)
                for scale, alphas in ((1.0, debug_alphas), (2.0, [8.0, 16.0, 32.0])):
                    d = direction * scale
                    for alpha in alphas:
                        steered = generate(
                            model, tokenizer, layers, device, PROBE_Q, d, alpha, layer
                        )
                        score = effect_score(baseline, steered, trait)
                        print(
                            f"[debug] {trait} L{layer} scale={scale} a={alpha} "
                            f"ok={score['ok']} diff={score['diff_chars']} markers={score['markers']}",
                            flush=True,
                        )
                        search_log.append(
                            {
                                "phase": "debug",
                                "trait": trait,
                                "layer": layer,
                                "alpha": alpha,
                                "scale": scale,
                                "score": score,
                            }
                        )
                        if score["ok"]:
                            key = (
                                score["markers"] - score.get("base_markers", 0),
                                score["diff_chars"],
                            )
                            entry = {
                                "phase": "debug",
                                "trait": trait,
                                "layer": layer,
                                "alpha": alpha,
                                "scale": scale,
                                "score": score,
                                "reply": steered,
                                "direction_scale": scale,
                            }
                            if best is None or key > best[0]:
                                best = (key, entry)
            if best is not None:
                found[trait] = best[1]
                print(
                    f"FOUND {trait} via debug @ L{best[1]['layer']} α={best[1]['alpha']} "
                    f"scale={best[1].get('scale', 1)}",
                    flush=True,
                )

    # Final six replies: regenerate with found settings (or best weak attempt)
    final = {"baseline": {"reply": baseline, "question": PROBE_Q}, "traits": {}}
    for trait in traits:
        if trait in found:
            cfg = found[trait]
            scale = float(cfg.get("direction_scale", cfg.get("scale", 1.0)))
            direction = vectors[trait][cfg["layer"]] * scale
            reply = generate(
                model,
                tokenizer,
                layers,
                device,
                PROBE_Q,
                direction,
                float(cfg["alpha"]),
                int(cfg["layer"]),
            )
            final["traits"][trait] = {
                "layer": cfg["layer"],
                "alpha": cfg["alpha"],
                "scale": scale,
                "phase": cfg["phase"],
                "score": effect_score(baseline, reply, trait),
                "reply": reply,
            }
        else:
            # last resort: largest-diff candidate from log is not stored with text;
            # try L15 α=8 anyway for visibility
            layer = default_layer
            alpha = 8.0
            reply = generate(
                model,
                tokenizer,
                layers,
                device,
                PROBE_Q,
                vectors[trait][layer],
                alpha,
                layer,
            )
            final["traits"][trait] = {
                "layer": layer,
                "alpha": alpha,
                "scale": 1.0,
                "phase": "fallback_unfound",
                "score": effect_score(baseline, reply, trait),
                "reply": reply,
            }

    report = {
        "model_id": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "n_layers": n_layers,
        "n_stmt_per_pole": N_STMT,
        "question": PROBE_Q,
        "system": SYSTEM,
        "found_traits": sorted(found.keys()),
        "missing_traits": [t for t in traits if t not in found],
        "search_log": search_log,
        "final": final,
        "vector_norms_L15": {
            t: float(vectors[t][default_layer].norm()) for t in traits
        },
    }
    OUT.write_text(json.dumps(report, indent=2))
    print("\n===== FINAL =====", flush=True)
    print("BASELINE:\n", baseline, "\n", flush=True)
    for trait, info in final["traits"].items():
        print(
            f"\n[{trait}] L{info['layer']} α={info['alpha']} scale={info['scale']} "
            f"phase={info['phase']} ok={info['score']['ok']}\n{info['reply']}\n",
            flush=True,
        )
    print("wrote", OUT, flush=True)
    ok_all = all(final["traits"][t]["score"]["ok"] for t in traits)
    print("OCEAN_DEMO_OK" if ok_all else "OCEAN_DEMO_PARTIAL", flush=True)
    return 0 if ok_all else 2


if __name__ == "__main__":
    sys.exit(main())
