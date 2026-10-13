import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import { fixtureEnv } from './wackmall-test-env.mjs';

const [binary, seed] = process.argv.slice(2);
assert(binary && seed, 'usage: node test-wackmall-cold-upload.mjs STAGING_BINARY SEED');
function run(enabled, generated = true, direct = false, fallback = '', guard = '') {
    const env = {...fixtureEnv(), LLAMA_EXPERT_FUSE_STAGING: '1', LLAMA_EXPERT_ASYNC_COLD_UPLOAD: enabled ? '1' : '0',
        GGML_SCHED_EXPERT_TIMING: '1', GGML_SCHED_EXPERT_INPUT_TIMING: '1', GGML_SCHED_PROFILE: '0',
        WACKMALL_TEST_TIMING_GENERATED: generated ? '1' : '0', WACKMALL_TEST_JOIN_INPUT: '1'};
    if (direct) env.GGML_CUDA_DISABLE_GRAPHS = '1';
    if (fallback === 'pageable') env.GGML_CUDA_NO_PINNED = '1';
    if (fallback === 'callback') env.WACKMALL_TEST_CALLBACK = '1';
    if (guard) env.WACKMALL_TEST_COLD_UPLOAD_GUARD = guard;
    const result = spawnSync(binary, fallback === 'S0' ? [seed, '0'] : [seed], {env, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024});
    assert.equal(result.status, 0, `${direct ? 'direct' : 'replay'} ${guard || fallback || 'route'}\n${result.stderr}`);
    assert.match(result.stdout, /PASS: one-token staging/);
    const rows = [...result.stderr.matchAll(/expert_join_input: (\{[^\n]*\})/g)].map(x => JSON.parse(x[1]));
    return {rows, stderr: result.stderr};
}
assert(run(false).rows.filter(x => x.root_op === 'MOE_COLD').every(x => x.route === 'fallback'));
for (const direct of [false, true]) {
    const {rows} = run(true, true, direct);
    const cold = rows.filter(x => x.root_op === 'MOE_COLD');
    assert.equal(cold.length, 16);
    assert(cold.every(x => x.route === 'cold_async'), 'missing guarded asynchronous cold-result upload');
    assert(cold.every(x => x.setter_calls === 0 && x.internal_wait_us === 0));
    const user = rows.filter(x => x.user_input);
    assert.equal(user.length, 8);
    assert(user.every(x => x.route === 'user' && x.setter_calls === 1), 'user inputs retain immediate synchronous copy');
}
assert.equal(run(true, false).rows.length, 0);
for (const fallback of ['S0', 'pageable', 'callback']) assert.equal(run(true, true, false, fallback).rows.length, 0);
for (const direct of [false, true]) {
    for (const guard of ['sync', 'reuse', 'reset', 'free', 'abort']) {
        const result = run(true, true, direct, '', guard);
        assert.match(result.stderr, new RegExp(`PASS: pending cold upload drained at ${guard}; source and uploaded bytes preserved`));
    }
}
console.log('PASS: default-off/fallback, strict decode, direct/replay, immediate user copies and pending-upload lifetime guards');
