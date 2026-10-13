#undef NDEBUG
#include <cassert>
#include <cstdlib>
#include <cstring>
#include "wackmall-test-env.h"

int main() {
    const char * outputs[] = {"STATS", "USAGE", "OBSERVE", "SIDECAR_IN", "SIDECAR_OUT", "PREDICT_LOG"};
    for (const char * output : outputs) {
        const std::string name = std::string("LLAMA_EXPERT_") + output;
        setenv(name.c_str(), "/user/private-data", 1);
    }
    setenv("WACKMALL_TEST_LLAMA_EXPERT_OBSERVE", "/tmp/test-observations", 1);
    wackmall_test_output_env();
    for (const char * output : outputs) {
        const std::string name = std::string("LLAMA_EXPERT_") + output;
        if (std::strcmp(output, "OBSERVE") == 0) {
            assert(std::strcmp(getenv(name.c_str()), "/tmp/test-observations") == 0);
        } else {
            assert(getenv(name.c_str()) == nullptr);
        }
    }
    unsetenv("WACKMALL_TEST_LLAMA_EXPERT_OBSERVE");
    wackmall_test_output_env();
    assert(getenv("LLAMA_EXPERT_OBSERVE") == nullptr);
}
