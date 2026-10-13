import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import { fixtureEnv } from './wackmall-test-env.mjs';

const [binary, seed] = process.argv.slice(2);
assert(binary && seed, 'usage: node test-wackmall-join-input.mjs STAGING_BINARY SEED');
function fixture(inputs, generated, timing = true, direct = false, fallback = '') {
    const env = {...fixtureEnv(), LLAMA_EXPERT_FUSE_STAGING: '1', GGML_SCHED_PROFILE: '0',
        GGML_SCHED_EXPERT_TIMING: timing ? '1' : '0', GGML_SCHED_EXPERT_INPUT_TIMING: inputs ? '1' : '0',
        WACKMALL_TEST_TIMING_GENERATED: generated ? '1' : '0', WACKMALL_TEST_JOIN_INPUT: '1'};
    if (direct) env.GGML_CUDA_DISABLE_GRAPHS = '1';
    if (fallback === 'pageable') env.GGML_CUDA_NO_PINNED = '1';
    if (fallback === 'callback') env.WACKMALL_TEST_CALLBACK = '1';
    const run = spawnSync(binary, fallback === 'S0' ? [seed, '0'] : [seed], {env, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024});
    assert.equal(run.status, 0, run.stderr);
    assert.match(run.stdout, /PASS: one-token staging/);
    return [...run.stderr.matchAll(/expert_join_input: (\{[^\n]*\})/g)].map(x => JSON.parse(x[1]));
}
assert.equal(fixture(false, true).length, 0, 'default-off silence');
assert.equal(fixture(true, false).length, 0, 'prompt exclusion');
assert.equal(fixture(true, true, false).length, 0, 'requires expert timing');
for (const direct of [false, true]) {
    const rows = fixture(true, true, true, direct);
    assert(rows.length >= 16, 'missing per-input attribution');
    const groups = new Map();
    for (const row of rows) {
        assert.equal(row.generated_decode, true);
        assert([0, 1].includes(row.layer));
        assert.equal(row.destination_backend, 'CUDA0');
        assert.equal(row.shape.length, 4);
        assert.equal(typeof row.source_name, 'string');
        assert.equal(typeof row.root_name, 'string');
        assert(row.root_op && row.type && row.source_buffer);
        assert(row.bytes > 0);
        for (const key of ['copy_us', 'wait_us', 'enqueue_us', 'internal_wait_us', 'setter_calls']) {
            assert(Number.isSafeInteger(row[key]) && row[key] >= 0, key);
        }
        assert(row.enqueue_us + row.internal_wait_us <= row.copy_us);
        if (!groups.has(`${row.graph}:${row.layer}`)) groups.set(`${row.graph}:${row.layer}`, []);
        groups.get(`${row.graph}:${row.layer}`).push(row);
    }
    assert.equal(groups.size, 16);
    const users = rows.filter(x => x.user_input);
    assert.equal(users.length, 8, 'shared user input copied once per graph');
    for (const user of users) {
        assert.equal(user.source_name, 'join_user_input');
        assert.equal(user.route, 'user');
        assert.equal(user.bytes, 32 * 4);
        assert.equal(user.setter_calls, 1, 'user inputs retain immediate synchronous setter');
    }
    for (const group of groups.values()) {
        const cold = group.filter(x => x.root_op === 'MOE_COLD');
        assert.equal(cold.length, 1, 'one cold-result upload per join');
        assert.equal(cold[0].bytes, 32 * 2 * 4);
        assert.equal(cold[0].route, 'fallback');
        assert.equal(cold[0].setter_calls, 1, 'must observe existing synchronous setter');
        assert.equal(cold[0].source_host, true);
        assert.equal(cold[0].user_input, false);
    }
}
for (const fallback of ['S0', 'pageable', 'callback']) assert.equal(fixture(true, true, true, false, fallback).length, 0, fallback);
console.log('PASS: join input identity, internal wait attribution, default-off and unsupported-path silence');
