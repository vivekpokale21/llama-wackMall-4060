import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import { fixtureEnv } from './wackmall-test-env.mjs';

const [binary, seed] = process.argv.slice(2);
assert(binary && seed, 'usage: node test-wackmall-hot-join.mjs STAGING_BINARY SEED');
function fixture(enabled, generated, direct = false, disabled = false, fallback = '') {
    const env = {...fixtureEnv(), LLAMA_EXPERT_FUSE_STAGING: '1', GGML_SCHED_PROFILE: '0',
        GGML_SCHED_EXPERT_TIMING: enabled ? '1' : '0', WACKMALL_TEST_TIMING_GENERATED: generated ? '1' : '0'};
    if (direct) env.GGML_CUDA_DISABLE_GRAPHS = '1';
    if (fallback === 'pageable') env.GGML_CUDA_NO_PINNED = '1';
    if (fallback === 'callback') env.WACKMALL_TEST_CALLBACK = '1';
    const run = spawnSync(binary, disabled ? [seed, '0'] : [seed], {env, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024});
    assert.equal(run.status, 0, run.stderr);
    assert.match(run.stdout, /PASS: one-token staging/);
    return [...run.stderr.matchAll(/expert_path_timing: (\{[^\n]*\})/g)].map(x => JSON.parse(x[1]));
}
assert.equal(fixture(false, true).length, 0, 'default-off must be silent');
assert.equal(fixture(true, false).length, 0, 'unattributed forwards must be excluded');
for (const direct of [false, true]) {
    const rows = fixture(true, true, direct);
    assert.equal(rows.length, 16, 'two hot/cold layers for each of eight one-token staged graphs');
    for (const row of rows) {
        assert.equal(row.generated_decode, true);
        assert.equal(row.ready, true, 'existing join must complete the timing events');
        assert([0, 1].includes(row.layer));
        assert.equal(typeof row.gpu_done_at_cpu, 'boolean');
        for (const key of ['hot_suffix_us', 'hot_to_join_us', 'cold_compute_us', 'join_input_us', 'join_wait_us', 'join_copy_us']) {
            assert(Number.isFinite(row[key]) && row[key] >= 0, key);
        }
        assert(row.join_wait_us + row.join_copy_us <= row.join_input_us);
    }
    assert.equal(new Set(rows.map(x => x.graph)).size, 8);
}
assert.equal(fixture(true, true, false, true).length, 0, 'S0 must be excluded');
assert.equal(fixture(true, true, false, false, 'pageable').length, 0, 'unsupported pageable staging must be excluded');
assert.equal(fixture(true, true, false, false, 'callback').length, 0, 'callback execution must be excluded');
console.log('PASS: default-off silence, strict attribution, prompt/S0 exclusion, direct/replay timing and numerical parity');
