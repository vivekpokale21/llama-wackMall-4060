import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import { fixtureEnv } from './wackmall-test-env.mjs';

const [binary, seed] = process.argv.slice(2);
assert(binary && seed, 'usage: node test-wackmall-dispatch-profile.mjs STAGING_BINARY SEED');
function fixture(profile) {
    const env = {...fixtureEnv(), GGML_SCHED_PROFILE: profile};
    delete env.GGML_CUDA_DISABLE_GRAPHS;
    const run = spawnSync(binary, [seed], {env, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024});
    assert.equal(run.status, 0, run.stderr);
    assert.match(run.stdout, /PASS: one-token staging/);
    return run.stderr;
}
const disabled = fixture('0');
assert.doesNotMatch(disabled, /cuda_dispatch_profile:/, 'disabled profiling must be silent');
const enabled = fixture('1');
const scheduler = [...enabled.matchAll(/sched_profile: (\{[^\n]*\})/g)].map(x => JSON.parse(x[1]));
const dispatch = [...enabled.matchAll(/cuda_dispatch_profile: (\{[^\n]*\})/g)].map(x => JSON.parse(x[1]));
assert(dispatch.length > 0, 'missing CUDA dispatch measurements');
const cuda = scheduler.filter(x => x.backend === 'CUDA0');
assert.equal(dispatch.length, cuda.length, 'one aggregate per CUDA scheduler graph');
let captures = 0, launches = 0, reuse = 0;
for (let i = 0; i < dispatch.length; ++i) {
    const d = dispatch[i], s = cuda[i];
    assert.equal(d.sched, s.sched);
    assert.equal(d.graph, s.graph);
    assert.equal(d.tokens, s.tokens);
    assert.equal(d.backend, s.backend);
    assert.equal(d.calls, s.backend_splits, 'counters must reset at scheduler graph boundary');
    assert.equal(d.calls, d.direct + d.captures + d.reuse);
    assert.equal(d.launches, d.captures + d.reuse);
    for (const key of ['dispatch_us', 'bookkeeping_us', 'evaluate_us', 'graph_api_us', 'launch_us', 'graphs_cached']) {
        assert(Number.isSafeInteger(d[key]) && d[key] >= 0, key);
    }
    assert(d.dispatch_us <= s.compute_us, 'CUDA dispatch is nested in scheduler submission');
    assert(d.bookkeeping_us + d.evaluate_us + d.graph_api_us + d.launch_us <= d.dispatch_us);
    captures += d.captures; launches += d.launches; reuse += d.reuse;
}
assert(captures > 0 && launches > 0 && reuse > 0, 'exercise capture and reused launch paths');
console.log(`PASS: disabled silence, graph attribution/reset, timing containment, capture/reuse (${dispatch.length} graphs)`);
