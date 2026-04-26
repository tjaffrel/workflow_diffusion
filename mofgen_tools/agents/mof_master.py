"""MOFMaster MCP tool wrapper."""

from __future__ import annotations

import re

from agents.mof_master import MOFMaster, MOFGenerationResponse

METALS = {
    "Zn": "zinc", "Zr": "zirconium", "Cu": "copper", "Fe": "iron",
    "Co": "cobalt", "Ni": "nickel", "Al": "aluminum", "Cr": "chromium",
    "Mg": "magnesium", "Ca": "calcium", "Ti": "titanium", "V": "vanadium",
    "Mn": "manganese", "Hf": "hafnium",
}


def generate_mof(
    request: str,
    provider: str = "openai",
) -> dict:
    """Invoke MOFMaster to translate a natural language request into MOF specs.

    Args:
        request: Natural language, e.g. "Generate 3 MOFs with Zr-based SBUs".
        provider: ``"openai"`` or ``"anthropic"``.

    Returns:
        Dict with generated structures, formulas, and metadata.
    """
    master = MOFMaster(provider=provider)

    intent = _parse_intent(request)
    if intent["mode"] == "metal_specific":
        response = master.generate_metal_specific_structures(
            metal=intent["metal"], count=intent["count"]
        )
    elif intent["mode"] == "composition_specific":
        response = master.generate_composition_specific_structures(
            composition=intent["composition"], count=intent["count"]
        )
    else:
        response = master.generate_basic_structures(count=intent["count"])

    return _response_to_dict(response)


def _parse_intent(request: str) -> dict:
    """Extract generation mode, count, and parameters from a request string."""
    request_lower = request.lower()

    count = 1
    count_match = re.search(r"(\d+)\s+mof", request_lower)
    if count_match:
        count = int(count_match.group(1))

    # Composition detection first (checked before metals so "composition
    # Zn:0.2, C:0.4" isn't misclassified as metal-specific).
    if "composition" in request_lower:
        composition = _parse_composition(request)
        if composition:
            return {
                "mode": "composition_specific",
                "composition": composition,
                "count": count,
            }

    # Metal detection: match element symbols at word boundaries (case-
    # sensitive) or full element names (case-insensitive).  This avoids
    # false positives like "diverse" → V or "all" → Al.
    for symbol, name in METALS.items():
        if re.search(rf"\b{symbol}\b", request) or name in request_lower:
            return {"mode": "metal_specific", "metal": symbol, "count": count}

    return {"mode": "basic", "count": count}


def _parse_composition(request: str) -> dict[str, float] | None:
    """Try to extract element:fraction pairs from a request string.

    Supports patterns like ``Zn:0.2, C:0.4`` or ``Zn 20%, C 40%``.
    Returns *None* if no composition could be parsed.
    """
    # Pattern: Element followed by colon/space then a number (optionally %)
    pairs = re.findall(
        r"([A-Z][a-z]?)\s*[:=]\s*(\d+(?:\.\d+)?)\s*%?", request,
    )
    if pairs:
        composition = {}
        for elem, value in pairs:
            val = float(value)
            if val > 1:
                val /= 100.0
            composition[elem] = val
        return composition
    return None


def _response_to_dict(response: MOFGenerationResponse) -> dict:
    return {
        "success_count": response.success_count,
        "generation_time_s": response.generation_time,
        "mode": response.mode_used.value,
        "structures": [
            {
                "formula": s.formula,
                "cif": s.cif_content,
                "metal_sbu": s.metal_sbu,
                "properties": s.properties,
            }
            for s in response.structures
        ],
    }
