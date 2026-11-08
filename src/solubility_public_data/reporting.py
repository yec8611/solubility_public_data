"""Portable training reports; PNG figures, XLSX tables and Markdown narrative."""
import json
from pathlib import Path
import pandas as pd


def export_tables(path, tables):
    # Runtime export is intentionally portable; repository workbooks were authored
    # and independently verified using the artifact-tool release builder.
    with pd.ExcelWriter(path, engine='xlsxwriter', engine_kwargs={'options': {'strings_to_formulas': False, 'strings_to_urls': False}}) as writer:
        for name, rows in tables.items():
            frame = pd.DataFrame(rows); frame.to_excel(writer, sheet_name=name[:31],index=False)
            sheet=writer.sheets[name[:31]];sheet.freeze_panes(1,0)


def report(directory, date, export_xlsx=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(directory); frame=pd.DataFrame(json.loads((out/f'metrics_{date}.json').read_text()))
    group=['mode','dataset','split','operating_point']
    selected=['average_precision','auroc','mcc','recall','precision','soluble_retention','accepted_insoluble_fraction','prevalence']
    if 'spearman' in frame:
        selected.append('spearman')
    summary=frame.groupby(group)[selected].agg(['mean','std','count']).reset_index()
    summary.columns=['_'.join(x).rstrip('_') if isinstance(x,tuple) else x for x in summary.columns]
    (out/f'summary_{date}.json').write_text(summary.to_json(orient='records',indent=2)+'\n')
    if export_xlsx:
        export_tables(out/f'results_{date}.xlsx', {'Metrics':frame.to_dict('records'),'Summary':summary.to_dict('records'),'Predictions':json.loads((out/f'predictions_{date}.json').read_text())})
    plot=summary[(summary.operating_point=='default') & summary.split.isin(['oof','holdout','external'])]
    fig,ax=plt.subplots(figsize=(9,max(4,len(plot)*.45+1.5)))
    ax.barh(range(len(plot)),plot.average_precision_mean,xerr=plot.average_precision_std.fillna(0),capsize=3,label='AP mean ± SD')
    ax.scatter(plot.prevalence_mean,range(len(plot)),color='black',marker='|',s=130,label='Positive prevalence',zorder=3)
    ax.set_yticks(range(len(plot)),[f'{r.mode} / {r.dataset} / {r.split}' for r in plot.itertuples()])
    ax.invert_yaxis();ax.legend(loc='lower right',fontsize=8)
    protocol=json.loads((out/f'protocol_{date}.json').read_text())
    title='Five distinct training/holdout partitions; mean ± SD'
    if protocol['epochs_capped_for_smoke']:
        title='SMOKE CHECK ONLY — epoch-capped fits; mean ± SD'
    ax.set_xlim(0,1);ax.set_xlabel('Insoluble average precision');ax.set_title(title,fontsize=11)
    fig.tight_layout();fig.savefig(out/f'performance_{date}.png',dpi=150);plt.close(fig)
    note='SMOKE VALIDATION ONLY: epoch-capped results are not scientific performance estimates.' if protocol['epochs_capped_for_smoke'] else 'Prespecified configuration; no selection on external labels.'
    (out/f'report_{date}.md').write_text(f'# Training report — {date}\n\n{note}\n\nTraining rows: {protocol["n"]}; insoluble: {protocol["n_insoluble"]}; groups: {protocol["n_groups"]}. Seeds: {protocol["seeds"]}.\n\nSee protocol JSON for features/architecture and fits JSON for parameter counts, epoch selection and split membership. OOF predictions come from three training-only folds; five holdouts differ. Random splits allow homologous groups to cross boundaries; grouped splits do not. Repeated holdout SD is descriptive, not an independent-replicate confidence interval. External panels stay fixed, evaluated by five separately trained models.\n\nThresholds use training OOF scores. OOF threshold metrics are apparent after threshold selection; holdout metrics are the assessment. Sigmoid scores are uncalibrated. A target OOF recall does not guarantee future insolubility rejection or an accepted-set contamination rate.\n\n![Performance](performance_{date}.png)\n')
