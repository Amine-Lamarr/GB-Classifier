from sklearn.model_selection import train_test_split , RandomizedSearchCV ,learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.metrics import precision_score , recall_score , confusion_matrix , accuracy_score , f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# handling data
data = pd.read_csv(r"C:\Users\lenovo\Downloads\Heart_Attack_Risk_Levels_Dataset.csv")
print(data.head())
print(data.shape)
print(data.describe())
print(data.info())
print(data.isnull().sum())
#any text features ?
feat = data.select_dtypes(include='object').columns
print("features non-numerical : \n" , feat) # we will work just on "Result" as a target
cols = data.shape[1]
print(data.columns)
x = data.iloc[ : , : cols-3]
y = data.iloc[ : , cols-2 : cols-1]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# finding outliers 
Q1 = x.quantile(0.25)
Q3 = x.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (x < lower_bound)|(x > upper_bound)
print("outliers : \n" , outliers.sum())
x = x[~(outliers).any(axis=1)]
y = y[~(outliers).any(axis=1)]
x = pd.DataFrame(x)
# scaling data
scaler = StandardScaler()
x = scaler.fit_transform(x)
# splitting data
x_train , x_test , y_train , y_test = train_test_split(x ,y , test_size=0.2 , random_state=23)

# searching for best parameters
params = {
    'n_estimators' : [200 , 400 , 600 , 800] , 
    'min_samples_split' : [5 , 10 , 15 ,20 , 25] , 
    'min_samples_leaf': [4 ,8 ,10 , 12 ,14] , 
    'max_features' : [0.2 , 0.4 , 0.5 , 0.6 ,'sqrt'] , 
    'max_depth' : [4 , 8 , 10 ,12 ,14 ] ,
    'subsample' : [0.8 ,0.5 , 0.7] , 
    'learning_rate' : [0.1 , 1 , 0.001 , 10], 
}
test = RandomizedSearchCV( GradientBoostingClassifier() , params , cv=3 , random_state=34)
test.fit(x_train , y_train)
bst_params = test.best_params_
bst_estim = test.best_estimator_
bst_score = test.best_score_
print("best params : \n" , bst_params ) #  {'subsample': 0.8, 'n_estimators': 800, 'min_samples_split': 25, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'max_depth': 14, 'learning_rate': 0.001}
print("best estimators  : \n" , bst_estim)
print(f"best score : {bst_score*100:.2f}%") # 97.94%

# model
model = GradientBoostingClassifier(subsample =  0.8, n_estimators = 800,
                                   min_samples_split =  25, min_samples_leaf= 10,
                                   max_features = 'sqrt',max_depth =  14,
                                   learning_rate= 0.001 , random_state=43)  
model.fit(x_train , y_train)
predictions = model.predict(x_test)

# results
score_train = model.score(x_train , y_train)
score_test = model.score(x_test , y_test)
precision = precision_score(y_test , predictions , average='macro')
recall = recall_score(y_test , predictions ,  average='macro')
f1 = f1_score(y_test , predictions , average='macro')
cm = confusion_matrix(y_test , predictions)
acc =  accuracy_score(y_test , predictions )

print(f"score (train) : {score_train*100:.2f}% ")
print(f"score (test) : {score_test*100:.2f}% ")
print(f"accuracy score : {acc*100:.2f}%")
print(f"recall score : {recall*100:.2f}%")
print(f"precision score : {precision*100:.2f}%")
print(f"f1 score : {f1*100:.2f}%")
print("confusion matrix : \n" , cm)

# feature importance
feat_im = model.feature_importances_
feature_names = data.columns[:cols-3]
plt.style.use("fivethirtyeight")
plt.barh(range(len(feat_im)) , feat_im ,label = 'feature importance' , color='purple')
plt.yticks(ticks=range(len(feat_im)), labels=feature_names)
plt.xlabel("importance")
plt.ylabel("features")
plt.title("the most important features")
plt.legend()
plt.show()

# confusion matrix
# import seaborn as sns
# plt.style.use("fivethirtyeight")
# sns.heatmap(cm , annot=True , fmt='d' , cmap='blues' , xticklabels=model.classes_ , yticklabels=model.classes_)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("confusion matrix")
# plt.show()

# learning curves
plt.style.use("fivethirtyeight")
train_size , train_score , test_score = learning_curve(GradientBoostingClassifier() , x_train , y_train , cv=5 , random_state=23)
train_mean = np.mean(train_score , axis=1)
train_std = np.std(train_score , axis=1)
test_mean = np.mean(test_score , axis=1)
test_std = np.std(test_score , axis=1)
plt.plot(train_size , train_mean , c = 'orange' , label = 'train score')
plt.fill_between(train_size , train_mean - train_std , train_mean + train_std , alpha = 0.9)
plt.plot(train_size , test_mean , c = 'r' , label = 'test score')
plt.fill_between(train_size , test_mean - test_std , test_mean + test_std , alpha = 0.9)
plt.xlabel("Training Set Size")
plt.ylabel("scores")
plt.title("learning curves")
plt.legend()
plt.grid()
plt.show()