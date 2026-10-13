import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import { fixtureEnv } from './wackmall-test-env.mjs';

const [binary, seed] = process.argv.slice(2);
assert(binary && seed, 'usage: node test-wackmall-fused-staging.mjs STAGING_BINARY SEED');
function fixture(fused, graphs = true, disabled = false, pageable = false, callback = false, profile = true) {
    const env = {...fixtureEnv(), GGML_SCHED_PROFILE: profile ? '1' : '0', LLAMA_EXPERT_FUSE_STAGING: fused ? '1' : '0'};
    if (!graphs) env.GGML_CUDA_DISABLE_GRAPHS = '1';
    if (pageable) env.GGML_CUDA_NO_PINNED = '1';
    if (callback) env.WACKMALL_TEST_CALLBACK = '1';
    const run = spawnSync(binary, disabled ? [seed, '0'] : [seed], {env, encoding: 'utf8', maxBuffer: 32*1024*1024});
    assert.equal(run.status, 0, run.stderr);
    assert.match(run.stdout, /PASS: one-token staging/);
    return {
        scheduler: [...run.stderr.matchAll(/sched_profile: (\{[^\n]*\})/g)].map(x => JSON.parse(x[1])).filter(x => x.backend === 'CUDA0'),
        fused: [...run.stderr.matchAll(/sched_fused_stage: (\{[^\n]*\})/g)].map(x => JSON.parse(x[1])),
    };
}
const control = fixture(false);
const fused = fixture(true);
assert.equal(fused.scheduler.length, control.scheduler.length);
assert(fused.scheduler.length > 0, 'missing scheduler submission counts');
for (let i = 0; i < control.scheduler.length; ++i) {
    const eligible = i >= 8 && i < 16; // two CPU-MoE layers, one-token staging-on graphs
    assert.equal(fused.scheduler[i].backend_splits, control.scheduler[i].backend_splits - (eligible ? 2 : 0),
        'fused staging must eliminate one CUDA dispatch per eligible layer');
}
assert.equal(fused.fused.length, 8, 'one aggregate for each reused one-token graph');
assert(fused.fused.every(x => x.layers === 2 && x.copies === 4 && x.wait_us >= 0));
const direct = fixture(true, false);
assert.equal(direct.fused.length, 8, 'graphs-disabled execution retains ready waits and numerical parity');
assert.equal(fixture(true, true, true).fused.length, 0, 'S0 bypass');
assert.equal(fixture(true, true, false, true).fused.length, 0, 'pageable destinations must fall back');
assert.equal(fixture(true, true, false, false, true).fused.length, 0, 'callback evaluation must fall back');
const quiet = fixture(true, true, false, false, false, false);
assert.equal(quiet.scheduler.length, 0, 'profiling-off execution must be silent');
assert.equal(quiet.fused.length, 0, 'profiling-off execution retains numerical parity without diagnostic rows');
console.log('PASS: fused dispatch reduction, changing-input parity, direct execution, S0, pageable/callback fallback and profiling-off parity');
