#!/usr/bin/env python3
import dataclasses
import importlib
import os
import sys


def load_output_class():
    try:
        mod = importlib.import_module("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
        return mod.Qwen2_5_VLCausalLMOutputWithPast, "installed transformers"
    except Exception as e_installed:
        repo_src = os.path.join(os.path.dirname(__file__), "..", "transformers", "src")
        repo_src = os.path.abspath(repo_src)
        if repo_src not in sys.path:
            sys.path.insert(0, repo_src)
        try:
            mod = importlib.import_module("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
            return mod.Qwen2_5_VLCausalLMOutputWithPast, "repo transformers/src"
        except Exception as e_repo:
            raise RuntimeError(
                "Could not import Qwen2_5_VLCausalLMOutputWithPast from installed package or repo source"
            ) from Exception(f"installed error: {e_installed}; repo error: {e_repo}")


def main():
    cls, source = load_output_class()
    print(f"import source: {source}")
    print(f"class identity: {cls.__module__}.{cls.__name__}")
    print("mro:", " -> ".join(c.__name__ for c in cls.__mro__))

    if dataclasses.is_dataclass(cls):
        print("dataclass fields:")
        for f in dataclasses.fields(cls):
            default = f.default if f.default is not dataclasses.MISSING else "<MISSING>"
            default_factory = (
                f.default_factory if hasattr(f, "default_factory") and f.default_factory is not dataclasses.MISSING else "<none>"
            )
            print(f" - {f.name}: type={getattr(f.type, '__name__', repr(f.type))}, default={default}, default_factory={default_factory}")
    else:
        print("dataclass fields: <not a dataclass>")

    annotations = getattr(cls, "__annotations__", {})
    print("annotations keys:", list(annotations.keys()))
    if annotations:
        print("annotations detail:")
        for k, v in annotations.items():
            print(f" - {k}: {v}")

    outputs = cls(loss=None, logits=None)
    print("outputs instance type:", type(outputs))
    print("hasattr(outputs, 'inputs_embeds'):", hasattr(outputs, "inputs_embeds"))
    print("hasattr(outputs, 'hidden_states'):", hasattr(outputs, "hidden_states"))
    # helper to safely format values
    def fmt(v, max_len=200):
        try:
            r = repr(v)
        except Exception:
            r = f"<unreprable {type(v)}>"
        if len(r) > max_len:
            r = r[: max_len - 3] + "..."
        return f"{type(v)} {r}"

    print("available keys:", list(outputs.keys()) if hasattr(outputs, "keys") else "<no keys>")

    # show instance dict / attributes
    print("\nclass vars:")
    for k, v in sorted(vars(cls).items()):
        print(f" - {k}: {fmt(v, 300)}")

    print("\nclass dir():")
    for name in sorted(dir(cls)):
        print(f" - {name}")

    print("\ninstance vars (vars(outputs) / __dict__):")
    if hasattr(outputs, "__dict__"):
        for k, v in sorted(outputs.__dict__.items()):
            print(f" - {k}: {fmt(v)}")
    else:
        print(" - <no __dict__>")

    print("\ninstance dir():")
    for name in sorted(dir(outputs)):
        print(f" - {name}")

    # if mapping-like, show items
    if hasattr(outputs, "items"):
        print("\noutputs.items():")
        try:
            for k, v in outputs.items():
                print(f" - {k}: {fmt(v)}")
        except Exception as e:
            print(f" - <error iterating items: {e}>")


if __name__ == "__main__":
    main()
