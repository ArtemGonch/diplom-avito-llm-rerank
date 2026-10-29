const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../apps/avito_user_view/annotation.html'),'utf8');
const ctx={};vm.createContext(ctx);vm.runInContext(source.slice(source.indexOf('function csvCell('),source.indexOf('function exportAnswers(')),ctx);
test('human comments with quotes and newlines remain one CSV field',()=>{
 assert.equal(ctx.csvCell('ясно, но "длинно"\nвторая строка'),'"ясно, но ""длинно""\nвторая строка"');
});
test('exported comments do not execute as spreadsheet formulas',()=>{
 for(const s of ['=1+1',' +SUM(A1)','-1','@test']) assert.ok(ctx.csvCell(s).startsWith('"\''));
});
test('missing annotation remains empty, not numeric zero',()=>{
 assert.equal(ctx.csvCell(undefined),'""');assert.equal(ctx.csvCell(0),'"0"');
});
