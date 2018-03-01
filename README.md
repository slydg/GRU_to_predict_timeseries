# GRU_to_predict_timeseries
本项目使用 pytorch 深度学习框架，模型各类参数可以参见代码注释。
数据处理流程：
1，./dataset/get_features_&_clean.py 用于获取特征并清洗数据
2，./dataset/get_labels_&_reshape.py 用于PCA降维并打上训练使用的标签
3，./GRU_model/gru.py 用于训练，测试并保存模型
4，./GRU_model/use_model.py 用于使用模型进行预测
5，./backtest/backtest.py 用于编写交易策略并回测，确定模型是否有效
