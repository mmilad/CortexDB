"""Unit tests for dataset discovery / scoring."""
from __future__ import annotations

import tempfile
import os

import pytest

from app.schemas.dataset import DatasetRecord
from app.schemas.discovery import DatasetDiscoverRequest
from app.services.dataset_match import discover_datasets
from app.store import SqliteStore


@pytest.fixture()
def store():
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        path = f.name
    s = SqliteStore(path)
    yield s
    s.close()
    os.unlink(path)


def _make_ds(key, description, caps=None, tags=None, llm_summary=None) -> DatasetRecord:
    return DatasetRecord(
        dataset_key=key,
        display_name=key,
        schema_version="v1",
        semantic_description=description,
        usage_guidance="use it",
        retrieval_capabilities=caps or [],
        capability_tags=tags or [],
        llm_summary=llm_summary,
    )


def _register(store, *records: DatasetRecord):
    for r in records:
        store.upsert_dataset(r.dataset_key, r.model_dump())


# ---------------------------------------------------------------------------
# Basic scoring
# ---------------------------------------------------------------------------

def test_token_overlap_ranks_higher(store):
    _register(
        store,
        _make_ds("incidents", "stores known incidents and outage events"),
        _make_ds("weather", "weather forecast data for regions"),
    )
    req = DatasetDiscoverRequest(intent="search for known incidents")
    resp = discover_datasets(req, store)
    assert resp.candidates[0].dataset.dataset_key == "incidents"


def test_capability_filter_excludes_missing(store):
    _register(
        store,
        _make_ds("vector_ds", "documents for search", caps=["vector"]),
        _make_ds("filter_ds", "metadata only", caps=["filter_only"]),
    )
    req = DatasetDiscoverRequest(intent="anything", required_capabilities=["vector"])
    resp = discover_datasets(req, store)
    keys = [c.dataset.dataset_key for c in resp.candidates]
    assert "vector_ds" in keys
    assert "filter_ds" not in keys


def test_tag_boost(store):
    _register(
        store,
        _make_ds("tagged", "some description", tags=["ops", "incident"]),
        _make_ds("untagged", "some description", tags=[]),
    )
    req = DatasetDiscoverRequest(intent="something", tag_filters=["incident"])
    resp = discover_datasets(req, store)
    assert resp.candidates[0].dataset.dataset_key == "tagged"


def test_empty_registry_suggests_create_new(store):
    req = DatasetDiscoverRequest(intent="store user feedback data")
    resp = discover_datasets(req, store)
    assert resp.recommended_action == "create_new"
    assert resp.suggested_blueprint is not None


def test_good_match_suggests_use_existing(store):
    _register(store, _make_ds("logs", "application log records for debugging and search"))
    req = DatasetDiscoverRequest(intent="application log records for debugging")
    resp = discover_datasets(req, store)
    assert resp.recommended_action == "use_existing"


def test_content_kind_boost(store):
    _register(
        store,
        _make_ds("doc_ds", "documentation pages", caps=["keyword"]),
        _make_ds("event_ds", "event stream data", caps=["filter_only"]),
    )
    # Give doc_ds the "documents" content_kind by patching the data
    data = store.get_dataset("doc_ds")
    data["content_kind"] = "documents"
    store.upsert_dataset("doc_ds", data)

    req = DatasetDiscoverRequest(intent="find documentation", content_kind="documents")
    resp = discover_datasets(req, store)
    assert resp.candidates[0].dataset.dataset_key == "doc_ds"


def test_blueprint_slug_is_stable_prefix(store):
    req = DatasetDiscoverRequest(intent="customer support tickets")
    resp = discover_datasets(req, store)
    assert resp.suggested_blueprint is not None
    key = resp.suggested_blueprint.suggested_dataset_key
    assert key.startswith("customer_support_tickets_")


def test_candidates_capped_at_20(store):
    for i in range(30):
        _register(store, _make_ds(f"ds{i}", "some data about things"))
    req = DatasetDiscoverRequest(intent="things")
    resp = discover_datasets(req, store)
    assert len(resp.candidates) <= 20
