from __future__ import annotations

from fastapi import APIRouter
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app import __version__
from app.api.router import api_router
from app.namespaces import create_namespace, create_subspace, list_namespaces, list_subspaces

router = APIRouter(tags=["namespaces"])


class CreateNamespaceRequest(BaseModel):
    namespace: str = Field(
        ...,
        description="Namespace name. Used as the path prefix and SQLite filename stem.",
        examples=["dev", "chat_project", "book_app"],
    )


class CreateSubspaceRequest(BaseModel):
    subspace: str = Field(
        ...,
        description="Subspace name. Creates one SQLite database inside the parent namespace.",
        examples=["dev", "prod", "agent_researcher"],
    )


class NamespaceInfo(BaseModel):
    namespace: str
    db_path: str
    api_root: str


class SubspaceInfo(BaseModel):
    namespace: str
    subspace: str
    db_path: str
    api_root: str


def _openapi_schema(server_url: str, label: str) -> dict:
    schema = get_openapi(
        title=f"CortexDB API ({label})",
        version=__version__,
        description=(
            "Scoped CortexDB API documentation. Operations in this Swagger UI "
            f"execute against {server_url}."
        ),
        routes=api_router.routes,
    )
    schema["servers"] = [{"url": server_url}]
    return schema


@router.post(
    "/create_namespace",
    response_model=NamespaceInfo,
    summary="Create a namespace-backed SQLite database",
    description=(
        "Creates an isolated SQLite database for a namespace. "
        "After creation, call endpoints under /{namespace}, for example "
        "/dev/tools or /chat_project/context/index."
    ),
)
def post_create_namespace(body: CreateNamespaceRequest) -> NamespaceInfo:
    path = create_namespace(body.namespace)
    return NamespaceInfo(
        namespace=body.namespace,
        db_path=str(path),
        api_root=f"/{body.namespace}",
    )


@router.post(
    "/{namespace}/new_subspace",
    response_model=SubspaceInfo,
    summary="Create a subspace database under a namespace",
    description=(
        "Creates one isolated SQLite database below a namespace. "
        "Subspaces are the maximum nesting level. After creation, call endpoints "
        "under /{namespace}/{subspace}, for example /book_app/dev/tools."
    ),
)
def post_create_subspace(namespace: str, body: CreateSubspaceRequest) -> SubspaceInfo:
    path = create_subspace(namespace, body.subspace)
    return SubspaceInfo(
        namespace=namespace,
        subspace=body.subspace,
        db_path=str(path),
        api_root=f"/{namespace}/{body.subspace}",
    )


@router.get(
    "/namespaces",
    response_model=list[str],
    summary="List known namespaces",
)
def get_namespaces() -> list[str]:
    return list_namespaces()


@router.get(
    "/{namespace}/subspaces",
    response_model=list[str],
    summary="List subspaces for a namespace",
)
def get_subspaces(namespace: str) -> list[str]:
    return list_subspaces(namespace)


@router.get(
    "/{namespace}/openapi.json",
    include_in_schema=False,
)
def get_namespace_openapi(namespace: str) -> dict:
    return _openapi_schema(f"/{namespace}", f"namespace: {namespace}")


@router.get(
    "/{namespace}/docs",
    include_in_schema=False,
)
def get_namespace_docs(namespace: str) -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url=f"/{namespace}/openapi.json",
        title=f"CortexDB API docs - {namespace}",
    )


@router.get(
    "/{namespace}/{subspace}/openapi.json",
    include_in_schema=False,
)
def get_subspace_openapi(namespace: str, subspace: str) -> dict:
    return _openapi_schema(
        f"/{namespace}/{subspace}",
        f"namespace: {namespace}, subspace: {subspace}",
    )


@router.get(
    "/{namespace}/{subspace}/docs",
    include_in_schema=False,
)
def get_subspace_docs(namespace: str, subspace: str) -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url=f"/{namespace}/{subspace}/openapi.json",
        title=f"CortexDB API docs - {namespace}/{subspace}",
    )
