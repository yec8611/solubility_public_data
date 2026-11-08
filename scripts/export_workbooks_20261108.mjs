// Release-authoring utility. Requires @oai/artifact-tool, supplied by the authoring
// environment; it is not a runtime dependency of the portable Python package.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {fileURLToPath} from 'node:url';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const args=process.argv.slice(2);
const specifications=[];
const rights={
  psi:'Attributed factual research extract from the NetSolP author release; no blanket commercial data license asserted. See DATA_LICENSES.md.',
  nesg:'Attributed factual research extract from the NetSolP/Price/NESG lineage; no blanket commercial data license asserted. See DATA_LICENSES.md.',
  garcia:'Garcia et al. 2026, Protein Science e70453; article/supplement CC BY 4.0; original-study terms retained.',
  esol:'eSOL © Hideki Taguchi (Tokyo Institute of Technology) licensed under CC Attribution-Share Alike 2.1 Japan. Adapted subset; fractions preserved.'
};
for (const name of ['psi','garcia','nesg','esol']) {
  const records=JSON.parse(await fs.readFile(path.join(root,`data/public/${name}_records_20261108.json`),'utf8'));
  specifications.push({file:`data/public/${name}_records_20261108.xlsx`,sheets:{Records:records,Sources:[{dataset:name,source_url:records[0].source_url,rights:rights[name],positive_class:'insoluble=1; soluble=0',changes:'Mini-protein filtering, family assignment, label inversion; eSOL strict binary extremes',created:'2026-11-08'}]}});
}
if(args[0]) {
  for(const architecture of ['residual_mlp','bilstm']) {
    const dir=path.join(args[0],architecture);
    const metrics=JSON.parse(await fs.readFile(path.join(dir,'metrics_20261108.json'),'utf8'));
    const summary=JSON.parse(await fs.readFile(path.join(dir,'summary_20261108.json'),'utf8'));
    specifications.push({file:`validation/${architecture}_smoke_20261108.xlsx`,sheets:{Notice:[{scope:'Two-epoch software smoke test; not scientific model performance',n:128,insoluble:64,seeds:'11,29,47,71,101',modes:'random and grouped',created:'2026-11-08'}],Metrics:metrics,Summary:summary}});
  }
}
for(const spec of specifications) {
  const wb=Workbook.create();
  for(const [name,rows] of Object.entries(spec.sheets)) {
    const sheet=wb.worksheets.add(name);const columns=[...new Set(rows.flatMap(r=>Object.keys(r)))];
    const clean=v=>v==null?null:(typeof v==='string' && /^[=+@]/.test(v)?"'"+v:v);
    const matrix=[columns,...rows.map(r=>columns.map(k=>clean(r[k])))];
    sheet.getRangeByIndexes(0,0,matrix.length,columns.length).values=matrix;
    sheet.getRangeByIndexes(0,0,1,columns.length).format.font.bold=true;
    sheet.getRangeByIndexes(0,0,matrix.length,columns.length).format.columnWidth=22;
    sheet.freezePanes.freezeRows(1);sheet.showGridLines=false;
    for(const [key,width] of [['record_id',38],['group_id',42],['sequence',115]])
      if(columns.includes(key))sheet.getRangeByIndexes(0,columns.indexOf(key),matrix.length,1).format.columnWidth=width;
  }
  wb.recalculate();
  const first=Object.keys(spec.sheets)[0];
  console.log((await wb.inspect({kind:'region',sheetId:first,range:'A1:F4',maxChars:700,tableMaxRows:4,tableMaxCols:6})).ndjson);
  const output=path.join(root,spec.file);await fs.mkdir(path.dirname(output),{recursive:true});
  const file=await SpreadsheetFile.exportXlsx(wb);await file.save(output);
  if(first==='Records'){
    const preview=await wb.render({sheetName:first,range:'A1:F6',scale:1,format:'png'});
    await fs.writeFile(path.join(os.tmpdir(),path.basename(spec.file,'.xlsx')+'_preview.png'),new Uint8Array(await preview.arrayBuffer()));
  }
  console.log(spec.file);
}
