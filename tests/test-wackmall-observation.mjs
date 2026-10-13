import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import { fixtureEnv, stagingEnv } from './wackmall-test-env.mjs';
import {mkdtempSync, readFileSync, existsSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
const [binary,seed] = process.argv.slice(2);
assert(binary && seed);
const root = mkdtempSync(join(tmpdir(),'wackmall-observation-'));
const path = join(root,'observations.jsonl');
const run = (extra={},disabled=false) => {
  const result = spawnSync(binary,[seed,...(disabled ? ['0'] : [])],{
    env:stagingEnv({...fixtureEnv(),LLAMA_EXPERT_OBSERVE:path,LLAMA_EXPERT_FUSE_STAGING:'1',...extra}),encoding:'utf8',timeout:180000,maxBuffer:32*1024*1024});
  assert.equal(result.status,0,result.stdout+result.stderr);
};
run();
assert(existsSync(path),'missing opt-in per-layer observation file');
const rows = readFileSync(path,'utf8').trim().split('\n').map(JSON.parse);
assert.equal(rows.length,64);
for (const row of rows) {
  assert([0,1].includes(row.layer),'actual tiered layer, never fully GPU layer');
  assert([1,4].includes(row.tokens),'no large prompt observations');
  assert.equal(row.total,2*row.tokens);
  assert.equal(row.hot,row.tokens);
  assert.equal(row.cold,row.tokens);
  assert.deepEqual(row.counts,[[0,row.tokens,1],[2,row.tokens,0]]);
}
assert.equal(rows.filter(row => row.tokens === 1).length,32);
assert.equal(rows.filter(row => row.tokens === 4).length,32);
const groups = Map.groupBy(rows,row => row.step);
assert.equal(groups.size,32);
for (const group of groups.values()) assert.deepEqual(group.map(row => row.layer),[0,1]);
const disabledPath = join(root,'disabled.jsonl');
run({LLAMA_EXPERT_OBSERVE:disabledPath},true);
assert(!existsSync(disabledPath),'S0 must not enable observations');
run({LLAMA_EXPERT_OBSERVE:''});
assert.equal(readFileSync(path,'utf8').trim().split('\n').length,64,'disabled observations must not append');
console.log('PASS: exact current counts, layer identity, reset/reuse, token metadata, S0 and disabled silence');
