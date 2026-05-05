"""Explicit helpers for reading and writing local artifacts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, NoReturn

import pandas as pd

_OBJECT_FORMATS_BY_SUFFIX = {
    ".pkl": "pickle",
    ".pickle": "pickle",
    ".json": "json",
    ".txt": "text",
}
_OBJECT_FORMAT_ALIASES = {
    "pkl": "pickle",
    "pickle": "pickle",
    "json": "json",
    "txt": "text",
    "text": "text",
}
_DATAFRAME_FORMATS_BY_SUFFIX = {
    ".parquet": "parquet",
    ".csv": "csv",
}
_DATAFRAME_FORMAT_ALIASES = {
    "parquet": "parquet",
    "csv": "csv",
}


def write_file(obj: Any, path: str | Path, *, format: str | None = None) -> Path:  # noqa: A002
    """Write a generic artifact object to an explicit path.

    Parameters
    ----------
    obj
        Object to persist.
    path
        Destination file path.
    format
        Optional object format override. Supported values are ``pickle``,
        ``json``, and ``text``.

    Returns
    -------
    Path
        Normalized path that was written.
    """
    output_path = _normalize_path(path)
    object_format = _resolve_format(
        output_path,
        supplied_format=format,
        suffix_formats=_OBJECT_FORMATS_BY_SUFFIX,
        format_aliases=_OBJECT_FORMAT_ALIASES,
        artifact_kind="object artifact",
    )
    _ensure_parent_dir(output_path)

    if object_format == "pickle":
        with output_path.open("wb") as file:
            pickle.dump(obj, file)
    elif object_format == "json":
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(obj, file, indent=2, sort_keys=True)
            file.write("\n")
    elif object_format == "text":
        with output_path.open("w", encoding="utf-8") as file:
            file.write(str(obj))
    else:  # pragma: no cover
        _raise_unsupported_format(object_format, "object artifact")

    return output_path


def read_file(path: str | Path, *, format: str | None = None) -> Any:  # noqa: A002
    """Read a generic artifact object from an explicit path.

    Parameters
    ----------
    path
        Source file path.
    format
        Optional object format override. Supported values are ``pickle``,
        ``json``, and ``text``.

    Returns
    -------
    Any
        Loaded artifact object.
    """
    input_path = _normalize_path(path)
    _ensure_file_exists(input_path)
    object_format = _resolve_format(
        input_path,
        supplied_format=format,
        suffix_formats=_OBJECT_FORMATS_BY_SUFFIX,
        format_aliases=_OBJECT_FORMAT_ALIASES,
        artifact_kind="object artifact",
    )

    if object_format == "pickle":
        with input_path.open("rb") as file:
            return pickle.load(file)  # noqa: S301
    if object_format == "json":
        with input_path.open(encoding="utf-8") as file:
            return json.load(file)
    if object_format == "text":
        with input_path.open(encoding="utf-8") as file:
            return file.read()

    return _raise_unsupported_format(object_format, "object artifact")


def write_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    **kwargs: Any,
) -> Path:
    """Write a pandas dataframe to an explicit path.

    Parameters
    ----------
    df
        Dataframe to persist.
    path
        Destination file path. Supported suffixes are ``.parquet`` and ``.csv``.
    index
        Whether to include the dataframe index.
    **kwargs
        Additional keyword arguments passed to the pandas writer.

    Returns
    -------
    Path
        Normalized path that was written.
    """
    output_path = _normalize_path(path)
    dataframe_format = _resolve_format(
        output_path,
        supplied_format=None,
        suffix_formats=_DATAFRAME_FORMATS_BY_SUFFIX,
        format_aliases=_DATAFRAME_FORMAT_ALIASES,
        artifact_kind="dataframe artifact",
    )
    _ensure_parent_dir(output_path)

    if dataframe_format == "parquet":
        df.to_parquet(output_path, index=index, **kwargs)
    elif dataframe_format == "csv":
        df.to_csv(output_path, index=index, **kwargs)
    else:  # pragma: no cover
        _raise_unsupported_format(dataframe_format, "dataframe artifact")

    return output_path


def read_dataframe(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a pandas dataframe from an explicit path.

    Parameters
    ----------
    path
        Source file path. Supported suffixes are ``.parquet`` and ``.csv``.
    **kwargs
        Additional keyword arguments passed to the pandas reader.

    Returns
    -------
    pandas.DataFrame
        Loaded dataframe.
    """
    input_path = _normalize_path(path)
    _ensure_file_exists(input_path)
    dataframe_format = _resolve_format(
        input_path,
        supplied_format=None,
        suffix_formats=_DATAFRAME_FORMATS_BY_SUFFIX,
        format_aliases=_DATAFRAME_FORMAT_ALIASES,
        artifact_kind="dataframe artifact",
    )

    if dataframe_format == "parquet":
        return pd.read_parquet(input_path, **kwargs)
    if dataframe_format == "csv":
        return pd.read_csv(input_path, **kwargs)

    return _raise_unsupported_format(dataframe_format, "dataframe artifact")


def _normalize_path(path: str | Path) -> Path:
    """Return a normalized absolute path without requiring the file to exist."""
    return Path(path).expanduser().resolve(strict=False)


def _ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for a write target."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_file_exists(path: Path) -> None:
    """Raise a path-specific error when a read target is missing."""
    if not path.exists():
        msg = f"Artifact file does not exist: {path}"
        raise FileNotFoundError(msg)
    if not path.is_file():
        msg = f"Artifact path is not a file: {path}"
        raise FileNotFoundError(msg)


def _resolve_format(
    path: Path,
    *,
    supplied_format: str | None,
    suffix_formats: dict[str, str],
    format_aliases: dict[str, str],
    artifact_kind: str,
) -> str:
    """Resolve a file format from an override or a path suffix."""
    if supplied_format is not None:
        normalized_format = supplied_format.lower().lstrip(".")
        if normalized_format in format_aliases:
            return format_aliases[normalized_format]
        _raise_unsupported_format(supplied_format, artifact_kind)

    suffix = path.suffix.lower()
    if suffix in suffix_formats:
        return suffix_formats[suffix]

    supported_suffixes = ", ".join(sorted(suffix_formats))
    if suffix:
        msg = f"Unsupported {artifact_kind} suffix for {path}: {suffix}. Supported suffixes: {supported_suffixes}."
    else:
        msg = f"Cannot infer {artifact_kind} format for {path}; supported suffixes: {supported_suffixes}."
    raise ValueError(msg)


def _raise_unsupported_format(format_name: str, artifact_kind: str) -> NoReturn:
    """Raise a consistent unsupported-format error."""
    msg = f"Unsupported {artifact_kind} format: {format_name}."
    raise ValueError(msg)
