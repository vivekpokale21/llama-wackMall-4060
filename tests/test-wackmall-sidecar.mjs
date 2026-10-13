import assert from 'node:assert/strict';
import { mkdtempSync, existsSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fixtureEnv, stagingEnv } from './wackmall-test-env.mjs';

const [binary, seed] = process.argv.slice(2).map(p => resolve(p));
assert(binary && seed, 'usage: node test-wackmall-sidecar.mjs BINARY SEED');
const dir = mkdtempSync(join(tmpdir(), 'wackmall-sidecar-'));
const output = join(dir, 'output.tier');
const env = { ...fixtureEnv(), LLAMA_EXPERT_FUSE_STAGING: '1',
    LLAMA_EXPERT_DECAY: '0.999', LLAMA_EXPERT_OBSERVE: '',
    WACKMALL_TEST_SIDECAR_LOAD: undefined, WACKMALL_TEST_GPU_LAYER: undefined,
    LLAMA_EXPERT_SIDECAR_IN: '', LLAMA_EXPERT_SIDECAR_OUT: output };
function run(overrides = {}, disabled = false) {
    const p = spawnSync(binary, [seed, ...(disabled ? ['0'] : [])], { env: stagingEnv({ ...env, ...overrides }), encoding: 'utf8', timeout: 180000, maxBuffer: 32 * 1024 * 1024 });
    assert.equal(p.status, 0, p.stderr);
    return p.stderr;
}
run();
assert(existsSync(output), 'missing explicit sidecar output');
const bytes = readFileSync(output);
assert.equal(bytes.length, 12 + 3 * 8 * 4, 'sidecar must include every model layer');
assert.equal(bytes.readUInt32LE(0), 1);
for (let il = 0; il < 3; il++) {
    const scores = Array.from({ length: 8 }, (_, e) => bytes.readFloatLE(12 + (il * 8 + e) * 4));
    assert(il === 2 ? scores.every(x => x === 0) : scores[0] > 0 && scores[1] > 0);
}
const restarted = join(dir, 'restarted.tier');
assert.match(run({ WACKMALL_TEST_SIDECAR_LOAD: '1', LLAMA_EXPERT_SIDECAR_IN: output,
    LLAMA_EXPERT_SIDECAR_OUT: restarted }), /loaded sidecar/);
assert(existsSync(restarted));
assert.deepEqual(readFileSync(output), bytes, 'input sidecar must remain immutable');
const hotLog = run({ LLAMA_EXPERT_SIDECAR_IN: output, LLAMA_EXPERT_SIDECAR_OUT: '' });
assert.match(hotLog, /LLAMA_EXPERT_HOT set, ignoring sidecar/);
assert.doesNotMatch(hotLog, /loaded sidecar|saved sidecar/);
const disabled = join(dir, 'disabled.tier');
run({ LLAMA_EXPERT_SIDECAR_OUT: disabled }, true);
assert(!existsSync(disabled), 'S0 must not save a sidecar');
const hole = join(dir, 'middle-gpu.tier');
run({ WACKMALL_TEST_GPU_LAYER: '1', LLAMA_EXPERT_SIDECAR_OUT: hole });
const holeBytes = readFileSync(hole);
assert.equal(holeBytes.length, bytes.length);
for (let il = 0; il < 3; il++) {
    assert.equal(holeBytes.readFloatLE(12 + il * 8 * 4) > 0, il !== 1, 'physical layer indexing');
}
assert.match(run({ WACKMALL_TEST_GPU_LAYER: '1', WACKMALL_TEST_SIDECAR_LOAD: '1',
    LLAMA_EXPERT_SIDECAR_IN: hole, LLAMA_EXPERT_SIDECAR_OUT: '' }), /loaded sidecar/);
assert(!existsSync(output + '.tmp') && !existsSync(hole + '.tmp'), 'atomic save completes');
console.log(`PASS: separate sidecar paths, mixed placement, restart parity and input immutability (${dir})`);
