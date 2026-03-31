import pytest
import torch

from k_language_model.rosa import ROSA, rosa_cpp_extension_available, rosa_next_token_ids, rosa_next_token_ids_batch


def _rosa_reference(x: list[int]) -> list[int]:
    n = len(x)
    y = [-1] * n
    s = 2 * n + 1
    b = [None] * s
    c = [-1] * s
    d = [0] * s
    e = [-1] * s
    b[0] = {}
    g = 0
    z = 1

    for i, t in enumerate(x):
        r = z
        z += 1
        b[r] = {}
        d[r] = d[g] + 1
        p = g
        while p != -1 and t not in b[p]:
            b[p][t] = r
            p = c[p]
        if p == -1:
            c[r] = 0
        else:
            q = b[p][t]
            if d[p] + 1 == d[q]:
                c[r] = q
            else:
                u = z
                z += 1
                b[u] = b[q].copy()
                d[u] = d[q] + 1
                c[u] = c[q]
                e[u] = e[q]
                while p != -1 and b[p][t] == q:
                    b[p][t] = u
                    p = c[p]
                c[q] = u
                c[r] = u
        v = g = r
        a = -1
        while v != -1:
            if d[v] > 0 and e[v] >= 0:
                a = x[e[v] + 1]
                break
            v = c[v]
        y[i] = a
        v = g
        while v != -1 and e[v] < i:
            e[v] = i
            v = c[v]
    return y


def test_rosa_matches_reference_on_known_sequences() -> None:
    sequences = [
        [],
        [7],
        [1, 2, 3, 4],
        [1, 2, 1, 2, 1, 2, 3],
        [5, 5, 5, 5, 5],
        [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],
    ]

    module = ROSA()

    for sequence in sequences:
        token_ids = torch.tensor(sequence, dtype=torch.int64)
        expected = torch.tensor(_rosa_reference(sequence), dtype=torch.int64)
        assert torch.equal(rosa_next_token_ids(token_ids), expected)
        assert torch.equal(module(token_ids), expected)


def test_rosa_returns_cpu_int64_tensor_with_same_length() -> None:
    token_ids = torch.tensor([4, 2, 4, 2, 8], dtype=torch.int32)
    output = ROSA()(token_ids)

    assert output.device.type == "cpu"
    assert output.dtype == torch.int64
    assert output.shape == token_ids.shape


def test_rosa_has_no_trainable_parameters() -> None:
    module = ROSA()

    assert list(module.parameters()) == []


def test_rosa_batch_exact_matches_reference_on_known_sequences() -> None:
    token_ids = torch.tensor(
        [
            [1, 2, 1, 2, 1, 2, 3],
            [5, 5, 5, 5, 5, 5, 5],
            [3, 1, 4, 1, 5, 9, 2],
        ],
        dtype=torch.int64,
    )
    expected = torch.tensor(
        [_rosa_reference(row.tolist()) for row in token_ids],
        dtype=torch.int64,
    )

    output = rosa_next_token_ids_batch(token_ids, impl="exact")

    assert torch.equal(output, expected)


def test_rosa_cpp_extension_probe_returns_bool() -> None:
    assert isinstance(rosa_cpp_extension_available(), bool)


def test_rosa_rejects_non_integer_or_non_1d_inputs() -> None:
    module = ROSA()

    with pytest.raises(TypeError):
        module(torch.tensor([1.0, 2.0], dtype=torch.float32))

    with pytest.raises(ValueError):
        module(torch.tensor([[1, 2], [3, 4]], dtype=torch.int64))
