"""Post-hoc explanation only: no fitting, model selection or threshold changes."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from finad_repair.hypergraph import feature_groups
from report_recent_zero import savecsv,fmt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--jobs',type=Path,required=True)
    p.add_argument('--verified',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert json.loads((a.verified/'verification.json').read_text())['status']=='passed'
    records=json.loads((a.verified/'records.json').read_text());rows=[]
    labels=dict(banksim=['金额','年龄','性别','交易类别','商户哈希'],s_ffsd=['金额','地点','类型','收款方哈希'])
    for task,names in labels.items():
        data=a.jobs/'himvh/inputs/tabular'/task;meta=json.loads((data/'manifest.json').read_text())
        groups=feature_groups(task,len(meta['features']));y=np.fromfile(data/'test.y.i8',np.int8);known=y!=-1
        for variant in ['m','cm']:
            for seed in [42,43,44]:
                f=a.jobs/'himvh/results/tabular'/task/'himvh'/variant/str(seed)
                with np.load(f/'components.npz',allow_pickle=False) as z:
                    means=z['calibration'].mean(0);flat=means*np.array([len(g) for g in groups]);comp=z['test']
                    for i,name in enumerate(names):
                        rows.append(dict(dataset=task,variant=variant,seed=seed,field=name,cal_mean_error=float(means[i]),
                                         macro_cal_share=float(means[i]/means.sum()),flat_cal_share=float(flat[i]/flat.sum()),
                                         test_field_AP=float(average_precision_score(y[known],comp[known,i])),exploratory_posthoc=True))
    savecsv(a.output/'posthoc_field_diagnostics.csv',rows)
    bounded=[]
    for r in records:
        if r['family']!='sdfmm':continue
        f=a.jobs/'sdfmm/results/tabular'/r['dataset']/'sdfmm'/r['variant']/str(r['seed'])
        threshold=json.loads((f/'thresholds.json').read_text())['pot'];op=r['metrics']['operating_points']['pot']
        bounded.append(dict(dataset=r['dataset'],variant=r['variant'],seed=r['seed'],threshold=threshold['threshold'],status=threshold['status'],
                            outside_probability_support=threshold['status']=='ok' and threshold['threshold']>1,
                            alarm_fraction=op['alarm_fraction'] if op else None,recall=op['recall'] if op else None,precision=op['precision'] if op else None))
    savecsv(a.output/'bounded_POT_diagnostics.csv',bounded)
    def ap(f,d,v):return np.mean([r['metrics']['average_precision'] for r in records if (r['family'],r['dataset'],r['variant'])==(f,d,v)])
    def field(task,v,name,key):return np.mean([r[key] for r in rows if (r['dataset'],r['variant'],r['field'])==(task,v,name)])
    def label_ap(f,d,v):return fmt(ap(f,d,v))+'%'
    text=['## 后验机制诊断：为什么修正仍未全面奏效','','本节在全部模型和分数固定、174配置独立复核通过后分析。字段AP使用测试标签作解释，**不是训练、选模或阈值依据**；不新增获胜方法、不替换预先固定主版本，也不改变370条主榜。字段分数是后验观察，不能解释为欺诈的因果因素。',
          '', '**本轮不支持直接把三个主修正版整体替换旧适配。** SD-FMM有明显数值改善，但HUGE／HIMVH没有一致收益；三个主修正版在候选池中均未取得最高均值AP或任一F1。修复目标／约束不一致，只是必要的机制检查，不是性能保证。',
          '', '**HUGE：角色重参数化有数据集代价，增加更新未补回。** BankSim的role主版AP为'+label_ap('huge','banksim','role')+'，旧复跑为'+label_ap('huge','banksim','legacy')+'，1200更新仍为'+label_ap('huge','banksim','role_long')+'；S-FFSD则从'+label_ap('huge','s_ffsd','legacy')+'升至'+label_ap('huge','s_ffsd','role')+'。这支持“本角色／缩放组合不具有普遍收益”，不支持“单纯多训练就能解决”。对齐、去角色列和分角色缩放是成组改变，尚不能单独断言哪个子环节抹掉了有用信号。',
          '', '**HIMVH：字段等权放大了不善于区分欺诈的误差，完整上下文也非必然有益。** BankSim的cm与cm_flat复用同一个检查点和同一批字段误差，仅换评分权重，AP由'+label_ap('himvh','banksim','cm')+'变为'+label_ap('himvh','banksim','cm_flat')+'；S-FFSD由'+label_ap('himvh','s_ffsd','cm')+'变为'+label_ap('himvh','s_ffsd','cm_flat')+'。BankSim的cm金额字段在校准平均总分中的贡献由按维数加权的'+fmt(field('banksim','cm','金额','flat_cal_share'))+'%升至等字段的'+fmt(field('banksim','cm','金额','macro_cal_share'))+'%，但金额单字段测试AP仅'+fmt(field('banksim','cm','金额','test_field_AP'))+'%；交易类别／商户字段的单字段AP更高，却被降权。等语义权重不等于等噪声尺度。',
          '', '同一新求解器与一致遮蔽下，BankSim由抽样上下文m的'+label_ap('himvh','banksim','m')+'变为完整上下文cm的'+label_ap('himvh','banksim','cm')+'；S-FFSD则由'+label_ap('himvh','s_ffsd','m')+'变为'+label_ap('himvh','s_ffsd','cm')+'。它测出了上下文效应的方向差异，但没有单独证明“过平滑”或某一种邻居类型是根因。BankSim的m_flat达到'+label_ap('himvh','banksim','m_flat')+'，属于消融证据，**不能事后替换预先指定主版本以宣称成功**。',
          '', '| 任务／固定检查点 | 字段 | 字段等权时校准贡献 | 按维数加权时校准贡献 | 单字段测试AP（后验） |','|---|---|---:|---:|---:|']
    for task,names in labels.items():
        for v in ['m','cm']:
            for name in names:text.append('| '+task+'/'+v+' | '+name+' | '+fmt(field(task,v,name,'macro_cal_share'))+' | '+fmt(field(task,v,name,'flat_cal_share'))+' | '+fmt(field(task,v,name,'test_field_AP'))+' |')
    previous=json.loads((Path(__file__).resolve().parents[1]/'results/recent_zero_20261117/per_seed.json').read_text())
    method=dict(huge='huge_financial_inductive',himvh='himvh_zero_reconstruction',sdfmm='sdfmm_zero_synthetic')
    drift=[abs(r['metrics']['average_precision']-next(v for v in previous if (v['dataset'],v['method'],v['seed'])==(r['dataset'],method[r['family']],r['seed']))['metrics']['average_precision']) for r in records if r['variant']=='legacy']
    assert len(drift)==30
    text+=['', '**SD-FMM：末点目标对齐最关键的证据出现在Sapienza；保持约束不代表人工异常更接近真实操纵。** Sapienza15S的e（在新公共RNG对照上只改末点目标）AP为'+label_ap('sdfmm','sapienza_15S','e')+'，再加入约束生成器ge为'+label_ap('sdfmm','sapienza_15S','ge')+'；25S由'+label_ap('sdfmm','sapienza_25S','e')+'变为'+label_ap('sdfmm','sapienza_25S','ge')+'。反之股票、加密货币和合成序列的G条件效应为正，说明生成器需要领域匹配，不能把修复时钟约束直接当作操纵语义匹配。',
           '', '**低F1与低AP需要分别解释。** '+str(sum(r['outside_probability_support'] for r in bounded))+'/108条SD-FMM配置出现POT阈值超过概率上界1的情况，本轮没有发现这一类越界问题。合成序列ge的平均测试报警率为'+fmt(np.mean([r['alarm_fraction'] for r in bounded if (r['dataset'],r['variant'])==('sd_synth','ge')]))+'%，异常召回率仅'+fmt(np.mean([r['recall'] for r in bounded if (r['dataset'],r['variant'])==('sd_synth','ge')]))+'%。因此AP提高没有转成好的固定操作点；不能据此把阈值改成测试最优。',
           '', '**旧法重跑也有波动。** 30条legacy复跑与历史报告的单种子AP最大差为'+fmt(max(drift))+'个百分点；输入与原始方法代码已核对，GPU训练不承诺位级确定。本轮修正效应统一对比当轮legacy，不拼接有利历史种子，详见[复跑偏差](legacy_reproduction.csv)。',
           '', '上述分量表只给3种子均值；完整逐种子字段文件见[后验字段诊断](posthoc_field_diagnostics.csv)，概率支持与报警率见[阈值诊断](bounded_POT_diagnostics.csv)。字段AP未参与任何拟合或主版本选择。下一步若改评分校准／生成器，需要另立方案并用新留出数据验证；本轮不在已查看测试集上继续调到获胜。','']
    (a.output/'posthoc_interpretation_CN.md').write_text('\n'.join(text))
    (a.output/'posthoc_diagnostic_scope.json').write_text(json.dumps(dict(exploratory_posthoc=True,additional_fits=0,train_labels_used=False,calibration_labels_used=False,test_labels_used_for_evaluation_only=True,primary_selection_changed=False,field_rows=len(rows),bounded_threshold_rows=len(bounded)),indent=2)+'\n')
    print('Exported54 posthoc field rows and108 bounded-score threshold checks; no training or selection changes')


if __name__=='__main__':main()
