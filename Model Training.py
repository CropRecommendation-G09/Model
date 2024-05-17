import xgboost as xgb

# 分离标签与特征值
x = df.drop(['label'], axis=1)  # 特征值
Y = df['label']  # 标签
labels = df['label'].tolist()
encode = preprocessing.LabelEncoder()
y = encode.fit_transform(Y)  # 标签编码

# 分离测试集以及数据集
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=10)

# 构建XGBoost模型
model = xgb.XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.1, random_state=5)
model.fit(x_train, y_train)

# 特征重要性分析
feature_names = x_test.columns
feature_importances = model.feature_importances_
indices = np.argsort(feature_importances)[::-1]
plt.figure()
plt.title("Feature Importance")
plt.bar(range(len(feature_importances)), feature_importances[indices], color='b')
plt.xticks(range(len(feature_importances)), np.array(feature_names)[indices], rotation=90)
plt.savefig('my_plot.png')

# 预测测试集
y_pred = model.predict(x_test)  # 定性数据
y_pred_quant = model.predict_proba(x_test)  # 定量数据

# 混淆矩阵
confusion_matrix_model = confusion_matrix(y_test, y_pred)

# 调用方法绘制混淆矩阵热力图
cnf_matrix_plotter(confusion_matrix_model, encode.classes_)  # 使用类别名进行可视化

# 将模型和标签编码器保存
joblib.dump(model, 'xgb_model.pkl')
joblib.dump(encoder, 'label_encoder.pkl')
