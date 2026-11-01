"""Convert explicitly selected local weights into a NInfer v3 artifact."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import re
import sys
from collections.abc import Mapping

from .official_recipes import RECIPES
from .pipeline import convert
from .proposal import DEFAULT_RANKING, add_official_proposal
from .qwen3_5 import build_model as build_qwen3_5
from .qwen4_exp import _ARCHITECTURES as QWEN4_EXP_ARCHITECTURES
from .qwen4_exp import build_model as build_qwen4_exp
from .qwen4_exp_gguf import RECIPES as QWEN4_EXP_GGUF_RECIPES
from .qwen4_exp_gguf import with_gguf_expert_count, with_mtp_expert_count
from .recipe import Recipe
from .sources.gguf import GGUFFile
from .sources.safetensors import SafetensorsSource
from .gguf_blocks import RECIPES as GGUF_RECIPES
from .gemma4 import ARCHITECTURES as GEMMA4_ARCHITECTURES
from .gemma4 import RECIPES as GEMMA4_RECIPES
from .gemma4 import build_model as build_gemma4
from .flint import ARCHITECTURES as FLINT_ARCHITECTURES
from .flint import RECIPES as FLINT_RECIPES
from .flint import build_model as build_flint
from .ternary import RECIPES as TERNARY_RECIPES


def _open_named_source(name: str, path: Path, block_sources: frozenset[str]):
    if path.suffix == ".gguf":
        # The ternary and GGUF block recipes read their block formats themselves, without gguf-py.
        if name in block_sources:
            return GGUFFile(path)
        if name == "dflash2":
            from .dflash2_gguf import DraftSource

            return DraftSource(path)
        from .sources.gguf_source import GGUFSource

        return GGUFSource(path)
    return SafetensorsSource(path)


class SourceInputs(Mapping):
    """Named optional sources are opened only when a recipe or component requests one."""

    def __init__(self, base, paths, stack, block_sources):
        self._sources = {"base": base}
        self._paths = dict(paths)
        self._stack = stack
        self._block_sources = block_sources

    def __getitem__(self, name):
        if name not in self._sources:
            if name not in self._paths:
                raise ValueError(
                    f"selected recipe requires source {name!r}; provide --source {name}=PATH"
                )
            self._sources[name] = self._stack.enter_context(
                _open_named_source(name, self._paths[name], self._block_sources)
            )
        return self._sources[name]

    def __iter__(self):
        return iter(dict.fromkeys((*self._sources, *self._paths)))

    def __len__(self):
        return len(set(self._sources) | set(self._paths))

    def provenance(self):
        return {
            name: {"path": str(source.path)} for name, source in self._sources.items()
        }


def _pairs(values, label):
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError(f"{label} requires unique NAME=PATH entries")
        result[name] = Path(path)
    return result


def _recipe_parts(value: str):
    """Split a FILE[:function] recipe reference.

    Only a colon followed by a bare identifier is the separator; any other colon -
    such as a Windows drive letter, or one inside a POSIX path component - belongs
    to the path."""
    filename, separator, function = value.rpartition(":")
    if not separator or not function.isidentifier():
        return value, "configure"
    return filename, function


def _layers(value: str):
    match = re.fullmatch(r"([0-9]+)\.\.([0-9]+)", value)
    if match is not None:
        begin, end = map(int, match.groups())
        if begin < end:
            return begin, end
    raise argparse.ArgumentTypeError("expected a zero-based range a..b with a < b; b is excluded")


def _function(value: str):
    recipes = {**RECIPES, **TERNARY_RECIPES, **GGUF_RECIPES, **QWEN4_EXP_GGUF_RECIPES,
               **GEMMA4_RECIPES, **FLINT_RECIPES}
    if value in recipes:
        return recipes[value]
    filename, function = _recipe_parts(value)
    path = Path(filename).resolve()
    spec = importlib.util.spec_from_file_location("ninfer_user_recipe", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load recipe file {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    result = getattr(module, function)
    if not callable(result):
        raise TypeError(f"{value}: recipe entry must be callable")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="primary checkpoint/config and default resources",
    )
    parser.add_argument(
        "--recipe",
        required=True,
        help="official name or Python file[:function]; defaults to configure in a file",
    )
    parser.add_argument(
        "--override",
        help="Python configuration applied after the selected recipe and proposal",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="named source such as quantized, dflash, dflash2, or mtp (replaces the base MTP head)",
    )
    parser.add_argument(
        "--components",
        help="comma-separated text,vision,mtp,dflash,dflash2 (default text); for "
        "Qwen3.8-Flash-Next text,ngram (the default: the model with its n-gram table), text (the "
        "model alone, its table read from a table artifact at run time) or ngram (that table "
        "artifact), and vision or mtp (from --source mtp=MTP.gguf or an HF safetensors subset) beside text",
    )
    parser.add_argument(
        "--resource",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="override a final frontend resource; Flash-Next also accepts ngram.hot=PROFILE "
        "to embed a hot-row profile with the stored n-gram table",
    )
    parser.add_argument(
        "--layers", type=_layers, metavar="A..B",
        help="Flash-Next qualification slice: source layers A through B-1, rebased to zero; "
        "retains embedding and head, omits the table when the slice has no PLE",
    )
    parser.add_argument(
        "--proposal",
        action="store_true",
        help="include the existing indexed proposal head",
    )
    parser.add_argument("--proposal-rows", type=int, default=131072)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--name", help="public instance name saved in metadata")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-chunk", type=int, default=512)
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        help="split the artifact into an entry and .part-NNNN files of at most this many bytes "
        "(default: one file)",
    )
    args = parser.parse_args(argv)
    paths = _pairs(args.source, "source")
    if "base" in paths:
        raise ValueError("select the base source with --model")
    overrides = _pairs(args.resource, "resource")
    with ExitStack() as stack:
        base = stack.enter_context(SafetensorsSource(args.model))
        architectures = base.config.get("architectures") or [None]
        flash_next = architectures[0] in QWEN4_EXP_ARCHITECTURES
        gemma4 = architectures[0] in GEMMA4_ARCHITECTURES
        flint = architectures[0] in FLINT_ARCHITECTURES
        if flint:
            if "gguf" not in paths or args.layers or (args.components or "text") != "text":
                raise ValueError("Flint converts its text component from --source gguf=FILE")
            gguf = stack.enter_context(GGUFFile(paths["gguf"]))
            model = build_flint(base, gguf, resource_overrides=overrides)
            recipe = Recipe(model)
            flint_sources = {"base": base, "gguf": gguf}
            _function(args.recipe)(model, recipe, flint_sources)
            if args.override:
                _function(args.override)(model, recipe, flint_sources)
            report = convert(
                model, recipe, args.out, name=args.name,
                provenance={"converter": "ninfer-v3", "recipe": args.recipe,
                            "sources": {"base": {"path": str(args.model)},
                                        "gguf": {"path": str(paths["gguf"])}}},
                device=args.device, rows_per_chunk=args.rows_per_chunk,
                max_file_bytes=args.max_file_bytes,
                progress=lambda index, total, job: print(
                    f"[{index+1}/{total}] {job.parameters[0]}"
                    + (f" (+{len(job.parameters)-1})" if len(job.parameters) > 1 else "")
                    + f": {job.spec.format} {job.spec.shape}", flush=True))
            print(f"wrote {args.out}: {report['objects']} objects, {len(report['files'])} files, "
                  f"{report['seconds']:.1f}s", flush=True)
            return
        if gemma4:
            wanted = (args.components or "text").split(",")
            if ("gguf" not in paths or args.layers or wanted[0] != "text"
                    or not set(wanted) <= {"text", "mtp", "vision"}
                    or len(set(wanted)) != len(wanted)
                    or ("mtp" in wanted) != ("mtp" in paths)
                    or ("vision" in wanted) != ("vision" in paths)):
                raise ValueError("Gemma 4 converts its text component from --source gguf=FILE; "
                                 "--components text,mtp adds its assistant from --source "
                                 "mtp=FILE and text,vision its vision tower from --source "
                                 "vision=MMPROJ.gguf")
            gguf = stack.enter_context(GGUFFile(paths["gguf"]))
            mtp = stack.enter_context(GGUFFile(paths["mtp"])) if "mtp" in paths else None
            vision = stack.enter_context(GGUFFile(paths["vision"])) if "vision" in paths else None
            model = build_gemma4(base, gguf, mtp=mtp, vision=vision,
                                 resource_overrides=overrides)
            recipe = Recipe(model)
            gemma_sources = ({"base": base, "gguf": gguf} | ({"mtp": mtp} if mtp else {})
                             | ({"vision": vision} if vision else {}))
            _function(args.recipe)(model, recipe, gemma_sources)
            if args.override:
                _function(args.override)(model, recipe, gemma_sources)
            source_paths = {"base": {"path": str(args.model)},
                            "gguf": {"path": str(paths["gguf"])}}
            for name in ("mtp", "vision"):
                if name in paths:
                    source_paths[name] = {"path": str(paths[name])}
            report = convert(
                model, recipe, args.out, name=args.name,
                provenance={"converter": "ninfer-v3", "recipe": args.recipe,
                            "sources": source_paths},
                device=args.device, rows_per_chunk=args.rows_per_chunk,
                max_file_bytes=args.max_file_bytes,
                progress=lambda index, total, job: print(
                    f"[{index+1}/{total}] {job.parameters[0]}"
                    + (f" (+{len(job.parameters)-1})" if len(job.parameters) > 1 else "")
                    + f": {job.spec.format} {job.spec.shape}", flush=True))
            print(f"wrote {args.out}: {report['objects']} objects, {len(report['files'])} files, "
                  f"{report['seconds']:.1f}s", flush=True)
            return
        if args.layers is not None and not flash_next:
            raise ValueError("--layers is implemented for Qwen3.8-Flash-Next")
        # Flash-Next's MTP GGUF is read block by block, as its model shard is.
        block_sources = frozenset(("ternary", "gguf", "ngram") + (("mtp",) if flash_next else ()))
        sources = SourceInputs(base, paths, stack, block_sources)
        if args.components is None:
            components = ("text", "ngram") if flash_next else ("text",)
        else:
            components = tuple(args.components.split(","))
        if len(components) != len(set(components)):
            raise ValueError("components must not repeat")
        companions = {
            key: sources[key] for key in ("dflash", "dflash2") if key in components
        }
        if "mtp" in components and "mtp" in paths:
            companions["mtp"] = sources["mtp"]
        build_model = build_qwen4_exp if flash_next else build_qwen3_5
        if build_model is build_qwen4_exp and "gguf" in paths:
            # An expert-pruned release takes the model's config with its own expert count.
            base.config = with_gguf_expert_count(base.config, paths["gguf"])
        if build_model is build_qwen4_exp and "mtp" in components and isinstance(
            companions.get("mtp"), GGUFFile
        ):
            # The MTP block keeps the experts its GGUF holds, whatever the model kept.
            base.config = with_mtp_expert_count(base.config, paths["mtp"])
        model = build_model(
            base,
            components=components,
            companions=companions,
            resource_overrides=overrides,
            **({"layers": args.layers} if flash_next else {}),
        )
        recipe = Recipe(model)
        _function(args.recipe)(model, recipe, sources)
        if args.proposal:
            add_official_proposal(recipe, ranking=args.ranking, rows=args.proposal_rows)
        if args.override:
            _function(args.override)(model, recipe, sources)

        def progress(index, total, job):
            label = job.parameters[0]
            if len(job.parameters) > 1:
                label += f" (+{len(job.parameters)-1})"
            print(
                f"[{index+1}/{total}] {label}: {job.spec.format} {job.spec.shape}",
                flush=True,
            )

        provenance = {
            "converter": "ninfer-v3",
            "recipe": args.recipe,
            "sources": sources.provenance(),
        }
        if args.override:
            provenance["override"] = args.override
        if args.layers is not None:
            provenance["source_layers"] = list(model.source_layers)
        if args.proposal:
            provenance["ranking"] = str(args.ranking)
        report = convert(
            model,
            recipe,
            args.out,
            name=args.name,
            provenance=provenance,
            device=args.device,
            rows_per_chunk=args.rows_per_chunk,
            max_file_bytes=args.max_file_bytes,
            progress=progress,
        )
        print(
            f"wrote {args.out}: {report['objects']} objects, {len(report['files'])} files, {report['seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
