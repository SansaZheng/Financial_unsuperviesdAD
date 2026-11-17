"""Numerical interpretation after all scores are locked; no model changes."""
import argparse,json
from pathlib import Path
import numpy as np
from report_recent_zero import readcsv,ms
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();summary=readcsv(a.results/'summary.csv');effects=readcsv(a.results/'paired_effects.csv');weights=json.loads((a.results/'weights.json').read_text());ix={(r['family'],r['dataset'],r['variant'],r['operating_point']):r for r in summary}
def delta(f,d,new,old):
 n=ix[f,d,new,'pot'];o=ix[f,d,old,'pot'];return f"{100*(n['average_precision_mean']-o['average_precision_mean']):+.2f}"
text=['### 根据本轮配对结果解释收益和失败','','以下解释在整轮分数和阈值固定、独立核验通过后生成，不参与权重或主版本选择。旧任务与新独立世界分别解释；未用新世界成绩继续修改方法。']
for d,name in [('banksim','BankSim旧任务'),('s_ffsd','S-FFSD旧任务'),('cardsim','CardSim独立模拟验证')]:
 text+=['','**'+name+'／HUGE：** 教师固定原40维几何、学生仍是相同48维时，AP变化'+delta('huge',d,'decoupled_embed','dual_embed')+'个百分点；将固定教师原始评分加入模型后变化'+delta('huge',d,'fusion','decoupled_embed')+'。但是fusion相对直接HALO变化'+delta('huge',d,'fusion','teacher_only')+'，这才说明训练分支是否提供超过教师本身的价值。']
 r=ix['huge',d,'fusion','pot'];t=ix['huge',d,'teacher_only','pot']
 if r['average_precision_mean']<=t['average_precision_mean']:text+=['本任务融合没有超过无需训练的直接教师。即使较旧HUGE提高，也不能把全部收益归功于新的学生训练；原几何信号或校准融合才可能是主要贡献，分支可能引入噪声。教师训练作用与推理直接教师作用已由两项独立对照分开。']
 else:text+=['本任务融合超过了直接教师均值，支持在本协议内存在互补信息；三种子差异不足以自动证明显著或跨来源稳定有效。']
 text+=['','**'+name+'／HIMVH：** 新主评分相对旧字段均值AP变化'+delta('himvh',d,'cm_skill_tail','cm_field')+'个百分点；相对已存在的维数加权尾证据仅变化'+delta('himvh',d,'cm_skill_tail','cm_tail_flat')+'。去维数先验的变化为'+delta('himvh',d,'cm_skill_equal_tail','cm_skill_tail')+'，去0.25权重下限为'+delta('himvh',d,'cm_skill_hard_tail','cm_skill_tail')+'。']
 if ix['himvh',d,'cm_skill_tail','pot']['average_precision_mean']<=ix['himvh',d,'cm_tail_flat','pot']['average_precision_mean']:text+=['重构增益权重在本任务没有超过已有宽度先验，不能把尾尺度或维数先验带来的提升宣传为skill权重收益。可预测性可能高估容易重构但与欺诈无关的字段，或低估真正异常但难以重构的字段；这仍是从受控结果提出的解释，不是已证明的普遍规律。']
 else:text+=['重构增益权重在本任务超过了相同尾尺度的宽度先验，增益可以与单纯尺度修正区分。它仍不等于已学到欺诈相关性，也须对照m、去宽度和去下限结果。']
 g=[r for r in weights if r['dataset']==d and r['base']=='cm'];sums=[sum(r['skill']>0 for r in g if r['seed']==s) for s in [42,43,44]];text+=['三个种子中具有正重构增益的字段数分别为'+str(sums)+'。逐字段权重可追溯，不是根据测试字段AP赋权。']
text+=['','仍未解决的限制：教师HALO反映原图统计异配而非确定的欺诈语义；尾证据会饱和并受校准分布影响；重构能力也不是异常判别能力。若独立模拟世界不支持旧任务收益，应报告泛化失败，不把该验证转为开发集再挑下一套权重。进一步真实来源或新模拟种子的确认实验需要单独预登记。','']
(a.results/'interpretation_CN.md').write_text('\n'.join(text))
