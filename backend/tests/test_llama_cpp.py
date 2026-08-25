from research.backends import LoadedModel, generate_gguf
from research.llama_cpp_setup import CUDA_INDEX, ensure_llama_cpp, install_commands


def test_install_commands_never_compile_from_sdist():
    cmds = install_commands()
    assert cmds
    binary_cmds = [c for c in cmds if "--only-binary=:all:" in c]
    assert binary_cmds, "CUDA/CPU index installs must refuse source builds"
    assert any(CUDA_INDEX.format(tag="cu124") in " ".join(c) for c in cmds)
    assert any("whl/cpu" in " ".join(c) for c in cmds)
    dry = ensure_llama_cpp(dry_run=True)
    assert dry["commands"] == cmds


def test_generate_gguf_prefers_usage_token_counts():
    class FakeLlama:
        def __call__(self, prompt, **kwargs):
            yield {"choices": [{"text": "Hello"}], "usage": {}}
            yield {
                "choices": [{"text": " world"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 4},
            }

        def tokenize(self, data, add_bos=True):
            raise AssertionError("usage counts should win over tokenize")

    loaded = LoadedModel(
        backend="llama.cpp",
        method="gguf",
        model_id="fake",
        device="cpu",
        precision="gguf_q4_k_m",
        load_time_s=0.0,
        _impl=FakeLlama(),
    )
    sample = generate_gguf(loaded, "hi", max_new_tokens=8)
    assert sample.prompt_tokens == 7
    assert sample.completion_tokens == 4
    assert sample.tokens_per_sec is not None
    assert "Hello world" in sample.text
