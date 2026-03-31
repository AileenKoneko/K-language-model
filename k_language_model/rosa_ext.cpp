#include <torch/extension.h>

#include <ATen/Parallel.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace {

torch::Tensor rosa_next_token_ids_batch_exact_cpu(torch::Tensor token_ids) {
    TORCH_CHECK(token_ids.device().is_cpu(), "ROSA C++ exact backend expects a CPU tensor.");
    TORCH_CHECK(token_ids.dim() == 2, "ROSA C++ exact backend expects a 2D tensor.");
    TORCH_CHECK(token_ids.scalar_type() == torch::kInt64, "ROSA C++ exact backend expects an int64 tensor.");

    auto input = token_ids.contiguous();
    const auto batch = input.size(0);
    const auto window = input.size(1);
    auto output = torch::full({batch, window}, -1, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    if (batch == 0 || window == 0) {
        return output;
    }

    auto in_acc = input.accessor<int64_t, 2>();
    auto out_acc = output.accessor<int64_t, 2>();

    at::parallel_for(0, batch, 1, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; ++b) {
            const int64_t n = window;
            const int64_t max_states = 2 * n + 1;

            std::vector<int64_t> sequence(n);
            for (int64_t i = 0; i < n; ++i) {
                sequence[i] = in_acc[b][i];
            }

            std::vector<std::unordered_map<int64_t, int64_t>> transitions(static_cast<size_t>(max_states));
            std::vector<int64_t> suffix_link(static_cast<size_t>(max_states), -1);
            std::vector<int64_t> depth(static_cast<size_t>(max_states), 0);
            std::vector<int64_t> end_pos(static_cast<size_t>(max_states), -1);

            int64_t last_state = 0;
            int64_t next_state = 1;

            for (int64_t i = 0; i < n; ++i) {
                const int64_t token = sequence[i];
                const int64_t current = next_state++;
                depth[current] = depth[last_state] + 1;
                int64_t state = last_state;

                while (state != -1) {
                    auto& table = transitions[state];
                    if (table.find(token) != table.end()) {
                        break;
                    }
                    table.emplace(token, current);
                    state = suffix_link[state];
                }

                if (state == -1) {
                    suffix_link[current] = 0;
                } else {
                    const int64_t target = transitions[state].at(token);
                    if (depth[state] + 1 == depth[target]) {
                        suffix_link[current] = target;
                    } else {
                        const int64_t clone = next_state++;
                        transitions[clone] = transitions[target];
                        depth[clone] = depth[target] + 1;
                        suffix_link[clone] = suffix_link[target];
                        end_pos[clone] = end_pos[target];

                        while (state != -1) {
                            auto it = transitions[state].find(token);
                            if (it == transitions[state].end() || it->second != target) {
                                break;
                            }
                            it->second = clone;
                            state = suffix_link[state];
                        }

                        suffix_link[target] = clone;
                        suffix_link[current] = clone;
                    }
                }

                last_state = current;
                state = last_state;
                int64_t predicted = -1;

                while (state != -1) {
                    if (depth[state] > 0 && end_pos[state] >= 0) {
                        const int64_t pred_idx = end_pos[state] + 1;
                        if (pred_idx < n) {
                            predicted = sequence[pred_idx];
                        }
                        break;
                    }
                    state = suffix_link[state];
                }

                out_acc[b][i] = predicted;
                state = last_state;
                while (state != -1 && end_pos[state] < i) {
                    end_pos[state] = i;
                    state = suffix_link[state];
                }
            }
        }
    });

    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rosa_next_token_ids_batch_exact_cpu",
        &rosa_next_token_ids_batch_exact_cpu,
        "Exact batched ROSA next-token ids on CPU");
}
