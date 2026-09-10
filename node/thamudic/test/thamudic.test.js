import test from 'node:test';import assert from 'node:assert/strict';import {isThamudic,extract} from '../src/index.js';
test('Thamudic Unicode range',()=>{assert.equal(isThamudic(0x10A80),true);assert.equal(isThamudic(65),false);assert.equal(extract('A𐪀B𐪁'),'𐪀𐪁');});
