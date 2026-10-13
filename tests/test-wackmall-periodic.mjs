import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fixtureEnv, stagingEnv } from './wackmall-test-env.mjs';
import { mkdtempSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const [binary, seed] = process.argv.slice(2);
assert(binary && seed);
const root = mkdtempSync(join(tmpdir(), 'wackmall-periodic-'));
function run(label, disabled = false, direct = false, extra = {}, seedPath = seed) {
    const env = { ...fixtureEnv(), WACKMALL_TEST_PERIODIC: '1', WACKMALL_TEST_OUTPUT: `${root}/${label}.bin`,
        LLAMA_EXPERT_PERIODIC: '1', LLAMA_EXPERT_FUSE_STAGING: '1', LLAMA_EXPERT_DECAY: '0.999',
        LLAMA_EXPERT_STATS: `${root}/${label}.stats`, LLAMA_EXPERT_OBSERVE: `${root}/${label}.jsonl`,
        LLAMA_EXPERT_SIDECAR_IN: '', LLAMA_EXPERT_SIDECAR_OUT: `${root}/${label}.tier`, ...extra };
    delete env.GGML_CUDA_GRAPH_OPT;
    if (direct) env.GGML_CUDA_DISABLE_GRAPHS = '1';
    else delete env.GGML_CUDA_DISABLE_GRAPHS;
    const p = spawnSync(binary, [seedPath, ...(disabled ? ['0'] : [])], { env: stagingEnv(env), encoding: 'utf8', timeout: 180000, maxBuffer: 32 * 1024 * 1024 });
    assert.equal(p.status, 0, p.stderr);
    writeFileSync(`${root}/${label}.log`, p.stderr);
    return readFileSync(`${root}/${label}.bin`);
}
const references = [];
for (let phase = 0; phase <= 2; phase++) {
    const csv = ['layer,expert,count', ...Array.from({ length: 3 }, (_, il) =>
        `${il},0,100\n${il},${il < phase ? 2 : 1},90`)].join('\n') + '\n';
    const path = `${root}/phase-${phase}.csv`;
    writeFileSync(path, csv);
    references.push(run(`reference-${phase}`, false, false,
        { WACKMALL_TEST_PERIODIC_NO_ADAPT: '1', WACKMALL_TEST_REFERENCE_PHASE: String(phase) }, path));
}
const reference = references[0];
for (const label of ['periodic', 'direct']) {
    const result = run(label, false, label === 'direct');
    assert.equal(result.length, reference.length);
    for (let i = 0; i < result.length; i += 4) {
        const step = Math.floor(i / (32 * 4));
        const phase = step < 2080 ? 0 : step < 2112 ? 1 : 2;
        assert(Math.abs(result.readFloatLE(i) - references[phase].readFloatLE(i)) <= 1e-6, `numeric parity ${label} ${i}`);
    }
    const stats = readFileSync(`${root}/${label}.stats`, 'utf8');
    assert.match(stats, /expert_tier: repins 2\b/);
    assert.match(stats, /sync_count 2147\b/, 'completion waits remain on every eligible graph');
    assert.match(stats, /decode_tokens 2115 boundaries 66 period 32 dwell 64 global_budget 1/);
    assert.doesNotMatch(stats, /INVARIANT VIOLATIONS/);
    const promotions = [...readFileSync(`${root}/${label}.log`, 'utf8').matchAll(/expert_promotion: (\{[^\n]*\})/g)].map(x => JSON.parse(x[1]));
    assert.equal(promotions.length, 2);
    for (let i = 0; i < 2; i++) {
        const p = promotions[i];
        assert.deepEqual([p.decode_tokens, p.boundary, p.layer, p.slot, p.old, p.expert, p.dwell, p.bytes],
            [2048 + i * 32, 64 + i, i, 1, 1, 2, 64 + i, 12288]);
        assert(p.new_score > p.old_score * 1.5 && p.copy_us >= 0);
    }
    const rows = readFileSync(`${root}/${label}.jsonl`, 'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(rows.length, 4294);
    assert(rows.every(row => row.total === 2));
    const bytes = readFileSync(`${root}/${label}.tier`);
    assert.equal(bytes.length, 108);
    for (let il = 0; il < 2; il++) {
        const score = e => bytes.readFloatLE(12 + (il * 8 + e) * 4);
        assert(score(2) > score(1) * 1.5);
        const decay = 0.999;
        const expectedCold = 32 * (1 - decay ** 66) / (1 - decay) + 3;
        assert(Math.abs(score(2) - expectedCold) < 0.02, 'decode-only window counts and final partial drain');
        assert(Math.abs(score(1) - 90 * decay ** 66) < 0.002, 'decay once per boundary, not per graph');
    }
}
for (const [label, extra] of [['off', { WACKMALL_TEST_PERIODIC_NO_ADAPT: '1' }],
    ['unattributed', { WACKMALL_TEST_PERIODIC_UNATTRIBUTED: '1' }]]) {
    const result = run(label, false, false, extra);
    for (let i = 0; i < result.length; i += 4) assert(Math.abs(result.readFloatLE(i) - reference.readFloatLE(i)) <= 1e-6);
    const stats = readFileSync(`${root}/${label}.stats`, 'utf8');
    assert.match(stats, /expert_tier: repins 0\b/);
    assert.match(stats, label === 'off' ? /decode_tokens 2115 boundaries 66/ : /decode_tokens 0 boundaries 0/);
    if (label === 'unattributed') assert(!existsSync(`${root}/${label}.tier`), 'unknown phase must not dirty/persist learned scores');
}
console.log(`PASS: 32-token boundaries, 64-opportunity dwell, one global promotion, reused/direct parity and clean lifetime (${root})`);
