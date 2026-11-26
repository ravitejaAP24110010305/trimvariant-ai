"""Feature catalogue and prompt construction."""
from __future__ import annotations

GLOBAL_NEGATIVE = (
    "blurry, deformed, distorted, warped, extra wheels, extra parts, text, watermark, logo, "
    "low quality, cartoon, illustration, artifacts, different car, inconsistent lighting"
)

STYLE_SUFFIX = "photorealistic automotive photograph, high detail, consistent lighting, reflections and shadows"

FEATURE_CATALOG: list[dict] = [
    {
        "id": "wheels", "label": "Wheels",
        "description": "Alloy wheel design or finish.",
        "template": "{option} on a car",
        "options": ["5-spoke silver alloy wheel", "multi-spoke 19-inch machined alloy wheel",
                    "gloss black split 10-spoke sport alloy wheel", "bronze turbine-style alloy wheel",
                    "chrome 6-spoke alloy wheel", "steel wheel with plastic hubcap"],
        "default_region": "auto_wheels",
    },
    {
        "id": "grille", "label": "Front grille",
        "description": "Grille pattern and finish.",
        "template": "{option} on the front of a car",
        "options": ["black honeycomb mesh grille", "chrome horizontal slat grille",
                    "piano-black grille with gloss trim", "sport mesh grille with red accent line"],
        "default_region": "front_grille",
    },
    {
        "id": "roof", "label": "Roof",
        "description": "Roof equipment or contrast roof.",
        "template": "{option} on a car roof",
        "options": ["panoramic glass sunroof", "black roof rails", "roof rack with crossbars",
                    "contrast gloss black roof"],
        "default_region": "side_roof",
    },
    {
        "id": "spoiler", "label": "Rear spoiler",
        "description": "Boot/trunk spoiler variants.",
        "template": "{option} on the rear of a car",
        "options": ["small body-colour trunk lip spoiler", "large sport rear wing spoiler",
                    "carbon-fibre ducktail spoiler"],
        "default_region": "rear_spoiler",
    },
    {
        "id": "headlights", "label": "Headlights",
        "description": "Headlamp signature and tint.",
        "template": "{option} on a car",
        "options": ["LED matrix headlights with signature daytime running lights",
                    "dark smoked headlights", "round projector headlamps"],
        "default_region": None,
    },
    {
        "id": "trim", "label": "Exterior trim",
        "description": "Window, door and body trim finish.",
        "template": "{option} on a car",
        "options": ["chrome window trim", "blacked-out window trim", "body-colour door handles",
                    "chrome side moulding"],
        "default_region": None,
    },
    {
        "id": "paint", "label": "Body colour",
        "description": "Repaint a body panel (draw the mask carefully).",
        "template": "{option} car body panel",
        "options": ["metallic red", "pearl white", "matte grey", "deep blue metallic", "British racing green"],
        "default_region": "side_door",
    },
    {
        "id": "custom", "label": "Custom feature",
        "description": "Describe any exterior feature yourself.",
        "template": "{option}",
        "options": [],
        "default_region": None,
    },
]

_BY_ID = {f["id"]: f for f in FEATURE_CATALOG}


def get_feature(feature_id: str) -> dict:
    if feature_id not in _BY_ID:
        raise ValueError(f"Unknown feature '{feature_id}'. Valid: {', '.join(_BY_ID)}")
    return _BY_ID[feature_id]


def build_prompt(feature_id: str, option: str | None, custom_prompt: str | None) -> str:
    feature = get_feature(feature_id)
    option = (option or "").strip()
    custom = (custom_prompt or "").strip()
    if feature_id == "custom":
        if not custom:
            raise ValueError("Custom feature needs a description in custom_prompt.")
        subject = custom
    else:
        if not option:
            option = feature["options"][0]
        subject = feature["template"].format(option=option)
        if custom:
            subject = f"{subject}, {custom}"
    return f"{subject}, {STYLE_SUFFIX}"


def build_negative(extra: str | None) -> str:
    extra = (extra or "").strip()
    return f"{GLOBAL_NEGATIVE}, {extra}" if extra else GLOBAL_NEGATIVE
