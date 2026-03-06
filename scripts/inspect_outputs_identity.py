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
        print("dataclass fields:", [f.name for f in dataclasses.fields(cls)])
    else:
        print("dataclass fields: <not a dataclass>")

    annotations = getattr(cls, "__annotations__", {})
    print("annotations keys:", list(annotations.keys()))

    outputs = cls(loss=None, logits=None)
    print("outputs instance type:", type(outputs))
    print("hasattr(outputs, 'inputs_embeds'):", hasattr(outputs, "inputs_embeds"))
    print("hasattr(outputs, 'hidden_states'):", hasattr(outputs, "hidden_states"))
    print("available keys:", list(outputs.keys()) if hasattr(outputs, "keys") else "<no keys>")


if __name__ == "__main__":
    main()
