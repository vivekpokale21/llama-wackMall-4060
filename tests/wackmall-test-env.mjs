export function fixtureEnv(inherited = process.env) {
    return Object.fromEntries(Object.entries(inherited).filter(([key]) =>
        !/^(LLAMA_EXPERT_|GGML_SCHED_|WACKMALL_TEST_)/.test(key) &&
        !['GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_GRAPH_OPT', 'GGML_CUDA_NO_PINNED'].includes(key)));
}

export function stagingEnv(env) {
    const result = { ...env };
    for (const output of ['STATS', 'USAGE', 'OBSERVE', 'SIDECAR_IN', 'SIDECAR_OUT', 'PREDICT_LOG']) {
        const name = `LLAMA_EXPERT_${output}`;
        if (Object.hasOwn(result, name)) {
            result[`WACKMALL_TEST_${name}`] = result[name];
            delete result[name];
        }
    }
    return result;
}
