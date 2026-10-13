#pragma once

#include <cmath>
#include <cstdint>

namespace llama_expert_tier {

inline bool server_generated_decode(int n_tokens, int n_slots, bool speculative, bool sampled_token) {
    return n_tokens == 1 && n_slots == 1 && !speculative && sampled_token;
}

struct promotion {
    int layer = -1, slot = -1, expert = -1;
    float old_score = 0.0f, new_score = 0.0f;
    uint64_t dwell = 0;
};

struct periodic_policy {
    uint64_t tokens = 0, opportunities = 0;
    static constexpr uint64_t period = 32, min_dwell = 64;
    static constexpr float hysteresis = 1.5f;

    // Unknown, prompt and multi-token graphs are not generated-decode evidence.
    bool advance(bool generated_decode, int64_t n_tokens) {
        if (!generated_decode || n_tokens != 1) {
            return false;
        }
        ++tokens;
        if (tokens % period != 0) {
            return false;
        }
        ++opportunities;
        return true;
    }

    bool eligible(const promotion & p) const {
        return p.layer >= 0 && p.slot >= 0 && p.expert >= 0 && p.dwell >= min_dwell &&
            std::isfinite(p.old_score) && std::isfinite(p.new_score) && p.old_score >= 0.0f &&
            p.new_score > hysteresis * p.old_score;
    }

    void consider(promotion & best, const promotion & p) const {
        if (!eligible(p)) {
            return;
        }
        const float gain = p.new_score - p.old_score;
        const float best_gain = best.new_score - best.old_score;
        if (best.layer < 0 || gain > best_gain ||
            (gain == best_gain && (p.layer < best.layer ||
                (p.layer == best.layer && (p.expert < best.expert ||
                    (p.expert == best.expert && p.slot < best.slot)))))) {
            best = p;
        }
    }
};

}
