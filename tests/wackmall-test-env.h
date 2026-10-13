#pragma once
#include <cstdlib>
#include <string>

inline void wackmall_test_output_env() {
    const char * outputs[] = {"STATS", "USAGE", "OBSERVE", "SIDECAR_IN", "SIDECAR_OUT", "PREDICT_LOG"};
    for (const char * output : outputs) {
        const std::string name = std::string("LLAMA_EXPERT_") + output;
        const std::string test_name = std::string("WACKMALL_TEST_") + name;
        unsetenv(name.c_str());
        if (const char * value = getenv(test_name.c_str())) {
            setenv(name.c_str(), value, 1);
        }
    }
}
