import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from  sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

data_train = r"E:\Projects & Tutorial\video tutorial\Machine learning\DATASETS\datasets\Mobile_train.csv"
data_test = r"E:\Projects & Tutorial\video tutorial\Machine learning\DATASETS\datasets\Mobile_test.csv"
data_train = pd.read_csv(data_train)
data_test = pd.read_csv(data_test)
data_test = data_test.drop('id', axis=1)

y = data_train['price_range']#dependent variable
x = data_train.drop('price_range', axis=1)#independent variable from training data

std = StandardScaler()
x_std = std.fit_transform(x)
# test_data_std = std.transform(data_test)

dtc = DecisionTreeClassifier()
# dtc.fit(x_std,y)
# predict_dataDTC=dtc.predict(test_data_std)
# scr_val=dtc.score(x_std,y)
# print(scr_val)
#===============KNN===============
knn = KNeighborsClassifier()
# knn.fit(x_std,y)
# predict_dataKNN = knn.predict(test_data_std)
# print(predict_dataKNN)
# print(knn.score(x_std,y))
#===============LogisticRegression===============

lr = LogisticRegression()
# lr.fit(x_std,y)
# predict_dataLogisticRegression=lr.predict(test_data_std)
# print(predict_dataLogisticRegression)
# print(lr.score(x_std,y))

#====================================================================

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=.2, random_state=None)

dtc.fit(x_train,y_train)
predict_valdtc=dtc.predict(x_test)
dtc_acc = accuracy_score(y_test,predict_valdtc)
# accuracy = int(accuracy)*100
# print(f"accuracy_score is {accuracy}")
#============================================================

x_train_std = std.fit_transform(x_train)
std_test_data = std.fit_transform(x_test)

knn.fit(x_train_std, y_train)
predict_valKNN=knn.predict(std_test_data)
Knn_acc = accuracy_score(y_test,predict_valKNN)
# print(accuracyknn)
# print(predict_valKNN)
#===============LR====================
lr.fit(x_train_std, y_train)
Y_lr_predicted=lr.predict(std_test_data)
lr_acc=accuracy_score(y_test,Y_lr_predicted)
confusion_matrix = confusion_matrix(y_test,Y_lr_predicted)

#==========SVM=================================
svc_moddel = SVC()
svc_moddel.fit(x_train_std,y_train)
y_svcPredicted = svc_moddel.predict(std_test_data)
svc_acc = accuracy_score(y_test,y_svcPredicted)
# print(svc_acc)
#=====================Naive Bays=============
gnb = GaussianNB()
gnb.fit(x_train_std, y_train)
y_gnb_predict = gnb.predict(std_test_data)
gnb_acc = accuracy_score(y_test, y_gnb_predict)
#========================RandomForestClassifier==================
rndForClf = RandomForestClassifier()
rndForClf.fit(x_train_std,y_train)
y_rndForClf_predict = rndForClf.predict(std_test_data)
rndForClf_acc = accuracy_score(y_test,y_rndForClf_predict)
# print(rndForClf_acc)
# print(confusion_matrix)
# total=(384)/(395)
# print(total)

#ploting the wholre scenario=================================
plt.bar(x=["dtc", "Knn","lr","SVC","gnb","rndForClf"], height=[dtc_acc,Knn_acc,lr_acc,svc_acc,gnb_acc,rndForClf_acc])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.show()


