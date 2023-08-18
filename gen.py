import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import xgboost as xgb
from sklearn.ensemble import VotingClassifier  


dataTrain=pickle.load(open(r'hogvectors_train.pkl','rb'))
dataTest=pickle.load(open(r'hogvectors_test.pkl','rb'))
xDataTrain=[]
yDataTrain=[]
xDdataTest=[]
yDataTest=[]

xDataTrain = [entry[:8100] for entry in dataTrain]
yDataTrain = [entry[8100] for entry in dataTrain]
label_encoder_train = LabelEncoder()
Y_numeric_train = label_encoder_train.fit_transform(yDataTrain)


xDdataTest = [entry[:8100] for entry in dataTest]
yDataTest = [entry[8100] for entry in dataTest]
label_encoder_test = LabelEncoder()
Y_numeric_test = label_encoder_test.fit_transform(yDataTest)


# X_train, X_test, y_train, y_test = train_test_split(X_data_train, Y_numeric_train, test_size=0.2)
model_tree = DecisionTreeClassifier(random_state=42)
model_xgb = xgb.XGBClassifier(objective="multi;softmax",num_class=len(Y_numeric_train),random_state=42)


# Decision tree and XGBoost together using Voting Classifier
ensemble_model = VotingClassifier(estimators=[('decision_tree', model_tree), ('xgb', model_xgb)], voting='hard',weights=[1,4])
ensemble_model.fit(xDataTrain, Y_numeric_train)
y_pred = ensemble_model.predict(xDdataTest)
accuracy = accuracy_score(Y_numeric_test, y_pred)
print("Ensemble Accuracy:", accuracy)
confusion_mat=confusion_matrix(Y_numeric_test,y_pred)
print(confusion_mat)

write_path_model="model.pkl"
pickle.dump(ensemble_model,open(write_path_model,"wb"))
print("done")
