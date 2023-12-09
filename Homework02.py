import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 데이터 불러오기
data_path = r"..\data\Catalunya Accidents data.csv"
data = pd.read_csv(data_path)
# 데이터 불러오기
data_path = r"..\data\Catalunya Accidents data.csv"
data = pd.read_csv(data_path)

# 데이터 전처리 과정
# 데이터 전처리 과정
# D_BOIRA
data['D_BOIRA'] = data['D_BOIRA'].map({"No n'hi ha": 0, "Si": 1})

# D_GRAVETAT
data['D_GRAVETAT'] = data['D_GRAVETAT'].map({'Accident greu': 1, 'Accident mortal': 2})

# D_CLIMATOLOGIA
data['D_CLIMATOLOGIA'] = data['D_CLIMATOLOGIA'].map({'Bon temps': 0, 'Pluja dèbil': 1})

# D_INFLUIT_VISIBILITAT
data['D_INFLUIT_VISIBILITAT'] = data['D_INFLUIT_VISIBILITAT'].map({'No': 0, 'Sense especificar': 1, 'Si': 2})

# D_INFLUIT_VISIBILITAT를 이진 분류로 변환 (예: 0 또는 1)
data['D_INFLUIT_VISIBILITAT'] = (data['D_INFLUIT_VISIBILITAT'] > 0).astype(int)

# 필요한 특징 선택 (피해자 수만 남김)
selected_features = ["F_VICTIMES", "C_VELOCITAT_VIA", "D_BOIRA", "D_GRAVETAT", "D_CLIMATOLOGIA", "D_INFLUIT_VISIBILITAT"]
data = data[selected_features]

# 데이터 전처리 - 필요한 특징 선택 및 누락된 값 처리
selected_features = ["F_VICTIMES", "C_VELOCITAT_VIA", "D_BOIRA", "D_GRAVETAT", "D_CLIMATOLOGIA", "D_INFLUIT_VISIBILITAT"]
data = data[selected_features]
data = data.fillna(data.mean())

# 필요한 특징 선택 (피해자 수만 남김)
selected_features = ["F_VICTIMES", "C_VELOCITAT_VIA", "D_BOIRA", "D_GRAVETAT", "D_CLIMATOLOGIA", "D_INFLUIT_VISIBILITAT"]
data = data[selected_features]

# 데이터 전처리 - 필요한 특징 선택 및 누락된 값 처리
data = data.fillna(data.mean())

# 특징과 레이블 분리
X = data.drop("F_VICTIMES", axis=1)
y = data["F_VICTIMES"]

# 학습용과 테스트용으로 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 구축 및 학습
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 테스트 데이터로 랜덤 포레스트 모델 평가
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"랜덤 포레스트 모델 정확도: {accuracy_rf}")

# 분류 보고서 출력
classification_rep_rf = classification_report(y_test, y_pred_rf, zero_division=1)
print("분류 보고서:\n", classification_rep_rf)

import matplotlib.pyplot as plt

# 특성 중요도 가져오기
feature_importances = rf_model.feature_importances_

# 특성 중요도를 내림차순으로 정렬
indices = sorted(range(len(feature_importances)), key=lambda k: feature_importances[k], reverse=True)

# 특성 이름 가져오기
feature_names = X.columns

# 시각화
plt.figure(figsize=(12, 8))
plt.bar(range(len(feature_importances)), feature_importances[indices], align="center")
plt.xticks(range(len(feature_importances)), [feature_names[i] for i in indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.show()

