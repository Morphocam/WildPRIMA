"""Which animal is this, and what horns does it have? A prior for the horn search, not a replacement for it.

The body fit itself stays species-blind. Species only decides which horns to try:
  * "paired": two horns, one each side of the skull (cattle, goats, antelope);
  * "nasal_two" / "nasal_one": horns on the midline of the nose, front one largest (rhinos);
  * "none": no horns (camels, horses, cats). "ossicones" (giraffes) are not modelled, so they also get none.

Camels also get their humps from it (two for Bactrian, one for dromedary), when the species is clear, and their
small ears: masks barely show ears, so a fitted ear only pads the outline, and SMAL's are a horse's.

The species comes from a label (`--species` or metadata.txt) when it names one species, otherwise from BioCLIP, a
zero-shot classifier for the tree of life, on crops of the clearest views. BioCLIP often cannot split close species
in small crops (black rhino, white rhino, hippo), but the horn layout is what matters, so probabilities are added up
per layout. An unknown or uncertain animal falls back to letting the masks decide, as before.

The table is hand-curated from the sources listed with it, rather than looked up online during a run: free text
does not give horn counts reliably, and a run should give the same answer offline, in the field, and next year.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

LAYOUTS = ("paired", "nasal_two", "nasal_one", "none")


@dataclass(frozen=True)
class Species:
    key: str
    common: str
    scientific: str
    horns: str  # one of LAYOUTS, or "ossicones"
    note: str
    source: str
    aliases: tuple[str, ...] = ()
    humps: int = 0
    ears: float | None = None  # ear length against SMAL's horse-like ears, when the species' are clearly different

    @property
    def layout(self) -> str:
        """The horns the body can build: ossicones are not modelled."""
        return "none" if self.horns == "ossicones" else self.horns


WIKI = "https://en.wikipedia.org/wiki/"
SPECIES = [
    Species("white_rhino", "white rhinoceros", "Ceratotherium simum", "nasal_two",
            "two horns on the nose, the front one larger (often 0.6 m), the rear one smaller",
            "https://www.savetherhino.org/rhino-info/species/white-rhinos/", ("white rhino", "southern white rhino")),
    Species("black_rhino", "black rhinoceros", "Diceros bicornis", "nasal_two",
            "two horns on the nose, front 0.5-1.3 m, rear up to about 0.5 m", WIKI + "Black_rhinoceros", ("black rhino",)),
    Species("sumatran_rhino", "Sumatran rhinoceros", "Dicerorhinus sumatrensis", "nasal_two",
            "two small horns; the only two-horned Asian rhino", WIKI + "Sumatran_rhinoceros", ("sumatran rhino",)),
    Species("indian_rhino", "Indian rhinoceros", "Rhinoceros unicornis", "nasal_one",
            "one horn on the nose, 20-60 cm", WIKI + "Indian_rhinoceros", ("indian rhino", "greater one-horned rhino")),
    Species("javan_rhino", "Javan rhinoceros", "Rhinoceros sondaicus", "nasal_one",
            "one horn on the nose, small", WIKI + "Javan_rhinoceros", ("javan rhino",)),
    Species("ankole", "Ankole-Watusi cattle", "Bos taurus", "paired", "very long horns, both sexes",
            WIKI + "Ankole-Watusi", ("ankle cow", "ankole", "ankole cow", "watusi")),
    Species("cattle", "domestic cattle", "Bos taurus", "paired", "paired horns, though many breeds are polled",
            WIKI + "Cattle", ("cow", "cattle", "bull")),
    Species("goat", "domestic goat", "Capra hircus", "paired", "paired horns in most breeds, both sexes",
            WIKI + "Goat", ("goat",)),
    Species("sheep", "domestic sheep", "Ovis aries", "paired", "rams of many breeds; ewes are often polled",
            WIKI + "Sheep", ("sheep",)),
    Species("oryx", "scimitar-horned oryx", "Oryx dammah", "paired", "long backward-curving horns, both sexes, up to 1.2 m",
            WIKI + "Scimitar_oryx", ("oryx", "scimitar oryx")),
    Species("roan", "roan antelope", "Hippotragus equinus", "paired", "backward-curving horns; females' shorter",
            WIKI + "Roan_antelope", ("roan",)),
    Species("wildebeest", "blue wildebeest", "Connochaetes taurinus", "paired", "curved horns, both sexes",
            WIKI + "Blue_wildebeest", ("wildebeest", "wildebest", "gnu")),
    Species("steenbok", "steenbok", "Raphicerus campestris", "paired", "short straight horns, males only",
            WIKI + "Steenbok", ("steenbok",)),
    Species("grysbok", "Cape grysbok", "Raphicerus melanotis", "paired", "short straight horns, males only",
            WIKI + "Cape_grysbok", ("grysbok",)),
    Species("springbok", "springbok", "Antidorcas marsupialis", "paired", "lyre-shaped horns, both sexes",
            WIKI + "Springbok", ("springbok",)),
    Species("gemsbok", "gemsbok", "Oryx gazella", "paired", "long straight horns, both sexes", WIKI + "Gemsbok", ("gemsbok",)),
    Species("giraffe", "giraffe", "Giraffa camelopardalis", "ossicones",
            "ossicones: skin-covered bony knobs, not modelled", WIKI + "Giraffe", ("giraffe",)),
    Species("bactrian_camel", "Bactrian camel", "Camelus bactrianus", "none", "no horns; two humps; small rounded ears",
            WIKI + "Bactrian_camel", ("bactrian camel",), humps=2, ears=0.45),
    Species("dromedary", "dromedary", "Camelus dromedarius", "none", "no horns; one hump; small rounded ears", WIKI + "Dromedary",
            ("dromedary", "arabian camel"), humps=1, ears=0.45),
    Species("horse", "horse", "Equus caballus", "none", "no horns", WIKI + "Horse", ("horse",)),
    Species("donkey", "donkey", "Equus asinus", "none", "no horns", WIKI + "Donkey", ("donkey",)),
    Species("zebra", "plains zebra", "Equus quagga", "none", "no horns", WIKI + "Plains_zebra", ("zebra",)),
    Species("hippo", "hippopotamus", "Hippopotamus amphibius", "none", "no horns", WIKI + "Hippopotamus", ("hippo",)),
    Species("elephant", "African elephant", "Loxodonta africana", "none", "tusks, not horns, and not modelled",
            WIKI + "African_elephant", ("elephant",)),
    Species("warthog", "common warthog", "Phacochoerus africanus", "none", "tusks, not horns, and not modelled",
            WIKI + "Common_warthog", ("warthog",)),
    Species("lion", "lion", "Panthera leo", "none", "no horns", WIKI + "Lion", ("lion", "lioness")),
    Species("tiger", "tiger", "Panthera tigris", "none", "no horns", WIKI + "Tiger", ("tiger",)),
    Species("cheetah", "cheetah", "Acinonyx jubatus", "none", "no horns", WIKI + "Cheetah", ("cheetah",)),
    Species("wolf", "grey wolf", "Canis lupus", "none", "no horns", WIKI + "Wolf", ("wolf",)),
]
BY_KEY = {s.key: s for s in SPECIES}
# Labels that name a group rather than one species: the classifier chooses within it.
GROUPS = {
    "rhino": ("white_rhino", "black_rhino", "sumatran_rhino", "indian_rhino", "javan_rhino"),
    "rhinoceros": ("white_rhino", "black_rhino", "sumatran_rhino", "indian_rhino", "javan_rhino"),
    "camel": ("bactrian_camel", "dromedary"),
    "antelope": ("oryx", "roan", "wildebeest", "steenbok", "grysbok", "springbok", "gemsbok"),
}
# A layout must have at least this share of the classifier's probability to be used.
MIN_LAYOUT_PROBABILITY = 0.5


def _normalise(label: str) -> str:
    label = re.sub(r"[_\-]+", " ", label.lower())
    return re.sub(r"\s*\d+$", "", label).strip()  # "Rhino_1" -> "rhino"


def candidates_for(label: str | None) -> tuple[str, ...]:
    """Species keys a label could mean: one, a group, or every species when the label is unknown or missing."""
    if label:
        name = _normalise(label)
        if name in GROUPS:
            return GROUPS[name]
        for s in SPECIES:
            if name in (s.key.replace("_", " "), s.common.lower(), s.scientific.lower(), *s.aliases):
                return (s.key,)
        singular = name[:-1] if name.endswith("s") else name
        if singular != name:
            return candidates_for(singular)
    return tuple(BY_KEY)


@dataclass
class HornDecision:
    layout: str | None  # a LAYOUT, or None: unknown, so the masks decide
    species: str | None  # species key when one is clear
    reason: str
    probabilities: dict
    humps: int = 0  # only from a clear species: a hump count is not guessed
    ears: float | None = None  # ear length when the species (or every species in its group) has a known one


def decide(label: str | None, probabilities: dict[str, float] | None) -> HornDecision:
    """Horn layout from a label and/or classifier probabilities over species keys (summing to 1)."""
    keys = candidates_for(label)
    if len(keys) == 1:
        s = BY_KEY[keys[0]]
        return HornDecision(s.layout, s.key, f"label '{label}' is {s.common}: {s.note}", probabilities or {}, s.humps, s.ears)
    if not probabilities:
        layouts = {BY_KEY[k].layout for k in keys}
        if len(keys) < len(BY_KEY) and len(layouts) == 1:  # a group that agrees, such as camels: no horns either way
            return HornDecision(layouts.pop(), None, f"label '{label}': every species in the group has the same horns", {},
                                ears=_shared_ears(keys))
        return HornDecision(None, None, f"label '{label}' is not one known species and no classifier ran; masks decide", {})
    within = {k: probabilities.get(k, 0.0) for k in keys}
    total = sum(within.values())
    if total <= 0:
        return HornDecision(None, None, "classifier gave no probability to the label's species; masks decide", probabilities)
    within = {k: v / total for k, v in within.items()}
    by_layout: dict[str, float] = {}
    for k, p in within.items():
        by_layout[BY_KEY[k].layout] = by_layout.get(BY_KEY[k].layout, 0.0) + p
    layout, share = max(by_layout.items(), key=lambda kv: kv[1])
    top = max(within, key=within.get)
    scope = f"within '{label}'" if label and len(keys) < len(BY_KEY) else "over all species"
    shares = ", ".join(f"{k} {v:.2f}" for k, v in sorted(by_layout.items(), key=lambda kv: -kv[1]))
    if share < MIN_LAYOUT_PROBABILITY:
        return HornDecision(None, None, f"classifier unsure {scope} ({shares}); masks decide", within)
    species = top if within[top] >= MIN_LAYOUT_PROBABILITY else None
    name = BY_KEY[top].common
    ears = BY_KEY[species].ears if species else _shared_ears([k for k in keys if BY_KEY[k].layout == layout])
    return HornDecision(layout, species, f"classifier {scope}: horn layouts {shares}; most likely {name} ({within[top]:.2f})", within,
                        BY_KEY[species].humps if species else 0, ears)


def _shared_ears(keys) -> float | None:
    """An ear length all these species share (both camels), or None."""
    ears = {BY_KEY[k].ears for k in keys}
    return ears.pop() if len(ears) == 1 else None


class SpeciesClassifier:
    """BioCLIP zero-shot over the species table, averaged over several crops."""

    MODEL = "hf-hub:imageomics/bioclip"

    def __init__(self, device="cuda"):
        import open_clip
        import torch

        self.torch, self.device = torch, device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.MODEL)
        self.model = self.model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(self.MODEL)
        prompts = [f"a photo of {s.scientific} ({s.common})" for s in SPECIES]
        with torch.no_grad():
            text = self.model.encode_text(tokenizer(prompts).to(device))
        self.text = text / text.norm(dim=-1, keepdim=True)

    def __call__(self, crops_rgb: list[np.ndarray]) -> dict[str, float]:
        from PIL import Image

        torch = self.torch
        images = torch.stack([self.preprocess(Image.fromarray(np.ascontiguousarray(c))) for c in crops_rgb]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(images)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            probs = (100.0 * emb @ self.text.T).softmax(dim=-1).mean(dim=0).cpu().numpy()
        return {s.key: float(p) for s, p in zip(SPECIES, probs)}
