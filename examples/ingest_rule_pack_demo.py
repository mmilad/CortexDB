"""Demo: teach CortexDB a deterministic ingest rule pack.

Run a CortexDB API first, then:
    python examples/ingest_rule_pack_demo.py
"""

from __future__ import annotations

import json

import httpx

BASE_URL = "http://127.0.0.1:8000"


def main() -> None:
    pack = {
        "key": "framework_knowledge",
        "display_name": "Framework Knowledge",
        "description": "Extract framework names and route them to framework memory.",
        "primitive_rules": [
            {
                "kind": "framework",
                "pattern": r"\b(Mastra|LangChain|LlamaIndex|Haystack)\b",
                "target_dataset_key": "frameworks",
                "confidence": 0.82,
                "metadata": {"domain": "agent_frameworks"},
            }
        ],
        "aliases": [
            {
                "canonical": "LangChain",
                "aliases": ["lang chain", "LCEL"],
                "kind": "framework_alias",
                "target_dataset_key": "frameworks",
            }
        ],
        "routing_hints": [
            {
                "target_dataset_key": "frameworks",
                "match_terms": ["agent framework", "RAG library"],
                "primitive_kinds": ["framework", "framework_alias"],
            }
        ],
        "examples": [
            {
                "label": "framework_mentions",
                "text": "Compare Mastra and LangChain for RAG routing.",
                "expected_primitives": [{"kind": "framework", "texts": ["Mastra", "LangChain"]}],
            }
        ],
    }

    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        guidance = client.get("/ingest/rule-packs/context").json()
        print("Accepted objects:", guidance["accepted_objects"])

        validation = client.post("/ingest/rule-packs/validate", json=pack).json()
        print("Validation:", json.dumps(validation, indent=2))
        if not validation["accepted"]:
            return

        stored = client.post("/ingest/rule-packs", json=pack).json()
        print("Stored:", stored["key"])

        analyzed = client.post(
            "/ingest/analyze",
            json={"text": "TODO: compare Mastra and LCEL as an agent framework."},
        ).json()
        print("Primitives:")
        for primitive in analyzed["primitives"]:
            print(f"- {primitive['kind']}: {primitive['text']} -> {primitive.get('target_dataset_key')}")


if __name__ == "__main__":
    main()
