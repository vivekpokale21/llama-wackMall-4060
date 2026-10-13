#include "llama-expert-policy.h"
#undef NDEBUG
#include <cassert>
#include <limits>

int main() {
    using namespace llama_expert_tier;
    periodic_policy policy;
    assert(!policy.advance(false, 1));
    assert(!policy.advance(true, 4));
    assert(policy.tokens == 0 && policy.opportunities == 0);
    for (int i = 1; i <= 2048; ++i) {
        assert(policy.advance(true, 1) == (i % 32 == 0));
    }
    assert(policy.opportunities == 64);
    promotion candidate {0, 1, 2, 90.0f, 150.0f, 63};
    assert(!policy.eligible(candidate));
    candidate.dwell = 64;
    assert(policy.eligible(candidate));
    candidate.new_score = 135.0f;
    assert(!policy.eligible(candidate));
    candidate.new_score = 150.0f;
    promotion best;
    policy.consider(best, candidate);
    promotion better {1, 1, 3, 10.0f, 160.0f, 64};
    policy.consider(best, better);
    assert(best.layer == 1);
    better.layer = 0;
    policy.consider(best, better);
    assert(best.layer == 0);
    better.new_score = std::numeric_limits<float>::quiet_NaN();
    assert(!policy.eligible(better));
    better.new_score = std::numeric_limits<float>::infinity();
    assert(!policy.eligible(better));
    better.new_score = 1.0f;
    better.old_score = 0.0f;
    assert(policy.eligible(better));
    return 0;
}
