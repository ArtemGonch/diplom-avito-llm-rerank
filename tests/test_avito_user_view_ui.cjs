/* Test the shipped browser functions against a minimal DOM, without networking. */
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../apps/avito_user_view/template.html'),'utf8');
function shipped(name,next){return source.slice(source.indexOf('function '+name+'('),source.indexOf('function '+next+'('));}
function el(tag,text){return {tag,textContent:text||'',attributes:{},children:[],classList:{add(){}},append(...xs){this.children.push(...xs)},setAttribute(k,v){this.attributes[k]=v},click(){this.clicked=true}};}
function context(extra={}){return {document:{createElementNS(_ns,tag){return el(tag)}},node:el,fmt:String,rub:v=>String(v),year:v=>String(Math.round(v)),...extra};}
for(const prices of [[1e6,1.1e6],[1e6,1e6],[2e5,2e7]]){
 test('price chart ticks and points share one log domain: '+prices.join('/'),()=>{
  const ctx=context();vm.createContext(ctx);vm.runInContext(shipped('priceChart','renderHistory'),ctx);
  const svg=ctx.priceChart({items:prices.map((price,i)=>({item:String(i),analyzed:true,price,year:2020+i,title:'Car'})),clusters:[]});
  const ticks=svg.children.filter(x=>x.tag==='text'&&x.textContent.endsWith('тыс. ₽'));
  const top=Number(ticks[0].textContent.split(' ')[0])*1000;
  const bottom=Number(ticks.at(-1).textContent.split(' ')[0])*1000;
  const circles=svg.children.filter(x=>x.tag==='circle');
  prices.forEach((price,i)=>{
   const expected=230-195*(Math.log10(price)-Math.log10(bottom))/(Math.log10(top)-Math.log10(bottom));
   assert.ok(Math.abs(circles[i].attributes.cy-expected)<1e-7);
  });
 });
}
test('summary download contains real line breaks',async()=>{
 let blob;const ctx=context({DATA:{period:['a','b']},Blob,URL:{createObjectURL(b){blob=b;return 'blob:test'},revokeObjectURL(){}}});
 vm.createContext(ctx);vm.runInContext(shipped('downloadSummary','summary'),ctx);
 ctx.downloadSummary({label:'User',facts:[{id:'f',text:'Grounded fact'}],summary:{sections:[{title:'Section',fact_ids:['f']}]}});
 const text=await blob.text();assert.ok(text.includes('\nSection\nGrounded fact\n'));assert.ok(!text.includes('\\n'));
});
