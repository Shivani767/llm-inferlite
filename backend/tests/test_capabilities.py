from research.capabilities import probe


def test_probe_has_required_keys():
    matrix = probe()
    assert "device" in matrix
    assert "experiments" in matrix
    for key in (
        "transformers_fp32",
        "int8_bnb",
        "int4_bnb",
        "awq",
        "gptq",
        "gguf",
        "tensorrt_llm",
        "smoothquant",
        "squeezellm",
        "paged_attention",
        "speculative_decoding",
        "continuous_batching",
    ):
        assert key in matrix["experiments"]
        item = matrix["experiments"][key]
        assert "supported" in item
        assert "reason" in item


def test_tensorrt_and_squeezellm_are_unsupported():
    matrix = probe()["experiments"]
    assert matrix["tensorrt_llm"]["supported"] is False
    assert matrix["squeezellm"]["supported"] is False
    assert matrix["smoothquant"]["supported"] is False
    assert "not" in matrix["tensorrt_llm"]["reason"].lower() or "Mac" in matrix["tensorrt_llm"]["reason"]
