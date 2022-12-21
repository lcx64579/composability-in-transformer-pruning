# 实验记录

原始未剪枝模型 旧称Baseline，现称Origin
Origin用朴素方法剪枝并微调 称为Baseline_pruned
块级微调 称Finetuned 实际上是Block-level Finetuned
模型级微调 称Retrained 实际上是Model-level Finetuned

|日期|文件名|阶段|剪枝率|Epochs|BLEU|备注|
|-|-|-|-|-|-|-|
|2022-10-19|baseline.pth|Origin|-|-|35.7072|Batch Size 128|
|2022-10-19|pruned_all_0_5.pth|Pruned|全部0.5|-|26.6144|-|
|2022-10-19|finetuned_all_0_5_epoch20.pth|Finetuned|全部0.5|20|29.8168|从epoch10开始，val_loss上升|
|2022-10-12|finetuned_all_0_5_epoch80.pth|Finetuned|全部0.5|80|21.6444|跟上一条比较，证明微调过拟合|
|2022-10-19|retrained_all_0_5_epoch20_epoch10.pth|Retrained|全部0.5|20 + 10|34.0671|-|
|2022-10-19|finetuned_all_0_5_epoch10.pth|Finetuned|全部0.5|10|30.8594|再finetune就过拟合了|
|2022-10-19|retrained_all_0_5_epoch10_epoch10.pth|Retrained|全部0.5|10 + 10|33.9951|-|
|2022-10-19|retrained_all_0_5_epoch10_epoch50.pth|Retrained|全部0.5|10 + 50|33.0263|-|
|-|-|-|-|-|-|-|
|2022-10-19|pruned_all_0_8.pth|Pruned|MHA0.75, Linear0.8|-|34.3056|注意0.8是指只剪掉了20%|
|2022-10-19|finetuned_all_0_8_epoch20.pth|Finetuned|MHA0.75, Linear0.8|20|33.7156|从epoch4开始val_loss上升。没有必要再retrain了|
|2022-10-19|finetuned_all_0_8_epoch4.pth|Finetuned|MHA0.75, Linear0.8|20|34.3266|恰在epoch4|
|-|-|-|-|-|-|-|
|2022-10-19|pruned_all_0_2.pth|Pruned|MHA0.25, Linear0.2|-|9.4030|注意0.2是指剪掉了80%|
|2022-10-19|finetuned_all_0_2_epoch11.pth|Finetuned|MHA0.25, Linear0.2|11/20|12.3924|总计20epoch，在11处最好（PS：什么最好？是BLEU还是valloss？）。这个是11|
|2022-10-19|retrained_all_0_2_epoch11_epoch10.pth|Retrained|MHA0.25, Linear0.2|11/20 + 10|34.4641|训练效果明显，虽然val loss还是没降[^1]|
|2022-10-19|retrained_all_0_2_epoch11_epoch30.pth|Retrained|MHA0.25, Linear0.2|11/20 + 30|32.4826|越retrain，BLEU越下降|
|2022-10-19|retrained_all_0_2_epoch11_epoch1.pth|Retrained|MHA0.25, Linear0.2|11/20 + 1|32.4670|-|
|2022-10-19|retrained_all_0_2_epoch11_epoch3.pth|Retrained|MHA0.25, Linear0.2|11/20 + 3|32.9233|最佳val loss，但并不是最佳BLEU|
|2022-11-16|retrained_all_0_2_epoch11_epoch4.pth|Retrained|MHA0.25, Linear0.2|11/20 + 4|33.9417|在4处最好。后面一直在29～32浮动。不再跟踪val loss|
|2022-11-16|pruned_all_0_2.pth -> retrained_all_0_2_epoch999.pth|Retrained|MHA0.25, Linear0.2|0 + 4|33.5361|没有finetune，改了个名字当做finetune过了|

[^1]: 剪枝率0.2（剪掉80%）的模型在retrain时，最初3个epochs有少许的val loss下降，然后上升，但幅度不大。其他的剪枝率均直接开始上升，严重过拟合。

有没有可能是：

- val loss函数不对？可能性低，教程用的就是交叉熵。但至少研究下。
- 块loss函数不对，不应用MSELoss？有可能，调查一下。
2022-11-09结论：既然val_loss和bleu之间不强相关，那就不看val_loss了。只看bleu。（2022-11-16：这件事只在retrain做了）
