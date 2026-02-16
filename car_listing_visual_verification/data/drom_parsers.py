from __future__ import annotations

from collections.abc import Iterable
import json
import re
from typing import Any
from urllib.parse import urljoin, urlparse

from selectolax.parser import HTMLParser

from car_listing_visual_verification.data.drom_utils import normalize_optional_str, stable_hash

JSON_ASSIGNMENT_PATTERNS = [
    re.compile(r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*;", re.DOTALL),
    re.compile(r"window\.__PRELOADED_STATE__\s*=\s*(\{.*?\})\s*;", re.DOTALL),
    re.compile(r"window\.__DATA__\s*=\s*(\{.*?\})\s*;", re.DOTALL),
]

IMAGE_EXT_RE = re.compile(r"\.(?:jpg|jpeg|png|webp|avif)(?:\?.*)?$", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _try_json_loads(text: str) -> Any | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _walk_json(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json(item)


def _flatten_scalar_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        candidate = normalize_optional_str(value)
        if candidate:
            yield candidate
        return

    if isinstance(value, (int, float)):
        yield str(value)
        return

    if isinstance(value, dict):
        for item in value.values():
            yield from _flatten_scalar_strings(item)
        return

    if isinstance(value, list):
        for item in value:
            yield from _flatten_scalar_strings(item)


def _looks_like_url(value: str) -> bool:
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        return True
    return value.startswith("/")


def _looks_like_listing_url(value: str, listing_patterns: list[re.Pattern[str]]) -> bool:
    if not _looks_like_url(value):
        return False
    return any(pattern.search(value) for pattern in listing_patterns)


def _looks_like_image_url(value: str) -> bool:
    if not _looks_like_url(value):
        return False
    return bool(IMAGE_EXT_RE.search(value))


def extract_json_blobs(content: str) -> list[Any]:
    blobs: list[Any] = []

    as_json = _try_json_loads(content)
    if as_json is not None:
        blobs.append(as_json)

    tree = HTMLParser(content)
    for script_node in tree.css("script"):
        script_text = script_node.text(strip=True)
        if not script_text:
            continue

        script_type = (script_node.attributes.get("type") or "").lower()
        if script_type in {"application/ld+json", "application/json"}:
            parsed = _try_json_loads(script_text)
            if parsed is not None:
                blobs.append(parsed)
            continue

        if script_node.attributes.get("id") in {"__NEXT_DATA__", "__NUXT_DATA__"}:
            parsed = _try_json_loads(script_text)
            if parsed is not None:
                blobs.append(parsed)
            continue

        for pattern in JSON_ASSIGNMENT_PATTERNS:
            for match in pattern.findall(script_text):
                parsed = _try_json_loads(match)
                if parsed is not None:
                    blobs.append(parsed)

    return blobs


def extract_listing_urls(
    content: str,
    base_url: str,
    listing_url_patterns: list[str],
) -> list[str]:
    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in listing_url_patterns]

    candidates: list[str] = []
    for blob in extract_json_blobs(content):
        for node in _walk_json(blob):
            if isinstance(node, dict):
                for key, value in node.items():
                    if "url" in key.lower():
                        for text in _flatten_scalar_strings(value):
                            candidates.append(text)
                    elif key.lower() in {"link", "links", "href", "path", "paths"}:
                        for text in _flatten_scalar_strings(value):
                            candidates.append(text)

    tree = HTMLParser(content)
    for anchor in tree.css("a[href]"):
        href = normalize_optional_str(anchor.attributes.get("href"))
        if href:
            candidates.append(href)

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not _looks_like_listing_url(candidate, compiled_patterns):
            continue
        full_url = urljoin(base_url, candidate)
        if full_url in seen:
            continue
        seen.add(full_url)
        normalized.append(full_url)

    return normalized


def extract_listing_id(url: str, listing_id_regex: str) -> str:
    match = re.search(listing_id_regex, url)
    if not match:
        return stable_hash(url)[:16]
    if match.groups():
        return match.group(1)
    return match.group(0)


def parse_listing_metadata(content: str, base_url: str) -> dict[str, Any]:
    blobs = extract_json_blobs(content)

    make_candidates: list[str] = []
    model_candidates: list[str] = []
    generation_candidates: list[str] = []
    year_candidates: list[int] = []
    image_candidates: list[str] = []

    make_keys = {"make", "brand", "manufacturer", "vendor"}
    model_keys = {"model", "serie", "series", "vehiclemodel"}
    generation_keys = {"generation", "gen", "bodycode"}
    year_keys = {
        "year",
        "modelyear",
        "manufactureyear",
        "productionyear",
        "buildyear",
    }

    image_keys = {
        "image",
        "images",
        "photo",
        "photos",
        "picture",
        "pictures",
        "gallery",
        "media",
    }

    for blob in blobs:
        for node in _walk_json(blob):
            if not isinstance(node, dict):
                continue

            for key, value in node.items():
                key_normalized = key.lower()
                scalar_values = list(_flatten_scalar_strings(value))

                if key_normalized in make_keys:
                    make_candidates.extend(scalar_values)
                elif key_normalized in model_keys:
                    model_candidates.extend(scalar_values)
                elif key_normalized in generation_keys:
                    generation_candidates.extend(scalar_values)
                elif key_normalized in year_keys:
                    for scalar in scalar_values:
                        if scalar.isdigit() and len(scalar) == 4:
                            year_candidates.append(int(scalar))
                elif key_normalized in image_keys or "image" in key_normalized:
                    for scalar in scalar_values:
                        if _looks_like_image_url(scalar):
                            image_candidates.append(scalar)

                if "url" in key_normalized or "href" in key_normalized:
                    for scalar in scalar_values:
                        if _looks_like_image_url(scalar):
                            image_candidates.append(scalar)

    tree = HTMLParser(content)
    title_node = tree.css_first("title")
    title = title_node.text(strip=True) if title_node else None

    h1_node = tree.css_first("h1")
    h1_text = h1_node.text(strip=True) if h1_node else None

    for selector in [
        "meta[property='og:image']",
        "meta[name='twitter:image']",
        "img[src]",
    ]:
        for node in tree.css(selector):
            candidate = node.attributes.get("content") or node.attributes.get("src")
            candidate = normalize_optional_str(candidate)
            if candidate and _looks_like_image_url(candidate):
                image_candidates.append(candidate)

    text_pool = " ".join(x for x in [title, h1_text] if x)
    for match in YEAR_RE.findall(text_pool):
        year_candidates.append(int(match))

    normalized_images: list[str] = []
    seen_images: set[str] = set()
    for url in image_candidates:
        normalized = urljoin(base_url, url)
        if normalized in seen_images:
            continue
        seen_images.add(normalized)
        normalized_images.append(normalized)

    return {
        "make": normalize_optional_str(make_candidates[0]) if make_candidates else None,
        "model": normalize_optional_str(model_candidates[0]) if model_candidates else None,
        "generation": normalize_optional_str(generation_candidates[0])
        if generation_candidates
        else None,
        "year": year_candidates[0] if year_candidates else None,
        "title": normalize_optional_str(title),
        "image_urls": normalized_images,
        "parser_blob_count": len(blobs),
    }
