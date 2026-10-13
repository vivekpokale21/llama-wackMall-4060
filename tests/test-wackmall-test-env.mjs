import assert from 'node:assert/strict';
import { fixtureEnv, stagingEnv } from './wackmall-test-env.mjs';

const inherited = {
    PATH: '/usr/bin', LD_LIBRARY_PATH: '/cuda/lib',
    LLAMA_EXPERT_SIDECAR_OUT: '/user/private-data', LLAMA_EXPERT_ASYNC_COLD_UPLOAD: '1',
    GGML_SCHED_PROFILE: '1', GGML_CUDA_NO_PINNED: '1', WACKMALL_TEST_PERIODIC: '1',
};
assert.deepEqual(fixtureEnv(inherited), { PATH: '/usr/bin', LD_LIBRARY_PATH: '/cuda/lib' });
const explicit = stagingEnv({ ...fixtureEnv(inherited), LLAMA_EXPERT_OBSERVE: '/tmp/test-output',
    LLAMA_EXPERT_FUSE_STAGING: '1', WACKMALL_TEST_ASYNC: '1' });
assert.equal(explicit.WACKMALL_TEST_LLAMA_EXPERT_OBSERVE, '/tmp/test-output');
assert.equal(explicit.LLAMA_EXPERT_OBSERVE, undefined);
assert.equal(explicit.LLAMA_EXPERT_FUSE_STAGING, '1');
assert.equal(explicit.WACKMALL_TEST_ASYNC, '1');
console.log('PASS: inherited controls removed; explicit test outputs preserved');
