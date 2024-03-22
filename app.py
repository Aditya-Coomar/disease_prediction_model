# Importing libraries
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
import streamlit_option_menu as option_menu
import uuid

st.set_page_config(page_title="Disease Prediction Model", page_icon="🧊", layout="wide", initial_sidebar_state="auto")

# ====================================================
# ROW 0 - LOAD DATASETS
#=====================================================
training_data = "dataset/Training.csv"
testing_data = "dataset/Testing.csv"

st.markdown('''<div style="font-size:70px; font-weight: bold;"> 
			Disease Prediction Model 
			</div>''',unsafe_allow_html=True)
st.divider()

st.markdown('''<div style="font-size:45px; font-weight: bold; padding-bottom: 20px"> 
			Model Characteristics 
			</div>''',unsafe_allow_html=True)
st.write()

# ====================================================
# ROW 1 - Data Testing
#=====================================================
row1_col1, row1_col2 = st.columns(2,gap="medium") 

with row1_col1 :
	# Reading the train.csv
	DATA_PATH = training_data
	try:
		data = pd.read_csv(DATA_PATH).dropna(axis = 1)
	except Exception as e :
		st.error("NOT_FOUND_ERROR - Could not find the training dataset. Check the path for the training dataset under section 'LOAD DATASETS'")
	
	# Checking whether the dataset is balanced or not
	disease_counts = data["prognosis"].value_counts()
	temp_df = pd.DataFrame({
        "Disease": disease_counts.index,
        "Counts": disease_counts.values
        })
	
	fig = plt.figure(figsize = (18,8))
	sns.barplot(x = "Disease", y = "Counts", data = temp_df)
	plt.xticks(rotation=90)
	st.pyplot(fig)

with row1_col2 :
	# Encoding the target value into numerical
	# value using LabelEncoder
	encoder = LabelEncoder()
	data["prognosis"] = encoder.fit_transform(data["prognosis"])
	
	X = data.iloc[:,:-1]
	y = data.iloc[:, -1]
	X_train, X_test, y_train, y_test =train_test_split(
		X, y, test_size = 0.2, random_state = 24)
	
	row1_sub1_col1, row1_sub1_col2 = st.columns(2)
	row1_sub1_col1.info(f"Train Data Size : {X_train.shape}, {y_train.shape}")
	row1_sub1_col2.info(f"Test Data Size : {X_test.shape}, {y_test.shape}")

	st.divider()

	row1_sub2_col1, row1_sub2_col2, row1_sub2_col3 = st.columns([1,2,1]) 
	
	# Defining scoring metric for k-fold cross validation
	def cv_scoring(estimator, X, y):
		return accuracy_score(y, estimator.predict(X))
	
	# Initializing Models
	models = {
		"SVC":SVC(),
		"Gaussian NB":GaussianNB(),
		"Random Forest":RandomForestClassifier(random_state=18)
		}
	
	# Producing cross validation score for the models
	for model_name in models:
		model = models[model_name]
		scores = cross_val_score(model, X, y, cv = 10,
						   n_jobs = -1,
						   scoring = cv_scoring)
		row1_sub2_col1.warning(model_name)
		row1_sub2_col2.success(f"Scores : {scores}")
		row1_sub2_col3.success(f"Mean Score : {np.mean(scores)}")


# ===========================================================
# ROW 2 - Model Accuracy Test : SVM & Naive Bayes Classifier
#============================================================
st.divider()
row2_col1, row2_col2 = st.columns(2,gap="large")

with row2_col1 :
	# Training and testing SVM Classifier
	svm_model = SVC()
	svm_model.fit(X_train, y_train)
	preds = svm_model.predict(X_test)

	st.info("Accuracy by Support Vector Machine Classifier")
	row2_sub1_col1, row2_sub1_col2 = st.columns(2)
	row2_sub1_col1.success(f"On Train Dataset\
		  : {accuracy_score(y_train, svm_model.predict(X_train))*100}")
	
	row2_sub1_col2.success(f"On Test Dataset\
		  : {accuracy_score(y_test, preds)*100}")
	cf_matrix = confusion_matrix(y_test, preds)
	svm_plot = plt.figure(figsize=(12,8))
	sns.heatmap(cf_matrix, annot=True)
	plt.title("Confusion Matrix for SVM Classifier on Test Data")
	st.pyplot(svm_plot)

with row2_col2 :
	# Training and testing Naive Bayes Classifier
	nb_model = GaussianNB()
	nb_model.fit(X_train, y_train)
	preds = nb_model.predict(X_test)
	st.info("Accuracy by Gaussian Naive Bayes Classifier")
	row2_sub2_col1, row2_sub2_col2 = st.columns(2)
	row2_sub2_col1.success(f"On Train Dataset\
		  : {accuracy_score(y_train, nb_model.predict(X_train))*100}")
	
	row2_sub2_col2.success(f"On Test Dataset\
		  : {accuracy_score(y_test, preds)*100}")
	cf_matrix = confusion_matrix(y_test, preds)
	nbc_plot = plt.figure(figsize=(12,8))
	sns.heatmap(cf_matrix, annot=True)
	plt.title("Confusion Matrix for Naive Bayes Classifier on Test Data")
	st.pyplot(nbc_plot)



# =========================================================================
# ROW 3 - Model Accuracy Test : Random Forest Classifier & Combined Models
#==========================================================================
st.divider()
row3_col1, row3_col2 = st.columns(2, gap="large")
with row3_col1 :
	# Training and testing Random Forest Classifier
	rf_model = RandomForestClassifier(random_state=18)
	rf_model.fit(X_train, y_train)
	preds = rf_model.predict(X_test)
	st.info("Accuracy by Random Forest Classifier")
	row3_sub1_col1, row3_sub1_col2 = st.columns(2)
	row3_sub1_col1.success(f"On Train Dataset\
		  : {accuracy_score(y_train, rf_model.predict(X_train))*100}")
	
	row3_sub1_col2.success(f"On Test Dataset\
		  : {accuracy_score(y_test, preds)*100}")
	
	cf_matrix = confusion_matrix(y_test, preds)
	rfc_plot = plt.figure(figsize=(12,8))
	sns.heatmap(cf_matrix, annot=True)
	plt.title("Confusion Matrix for Random Forest Classifier on Test Data")
	st.pyplot(rfc_plot)

with row3_col2 :
	# Training the models on whole data
	final_svm_model = SVC()
	final_nb_model = GaussianNB()
	final_rf_model = RandomForestClassifier(random_state=18)
	final_svm_model.fit(X, y)
	final_nb_model.fit(X, y)
	final_rf_model.fit(X, y)
	
	# Reading the test data
	try:
		test_data = pd.read_csv(testing_data).dropna(axis=1)
	except:
		st.error("NOT_FOUND_ERROR - Could not find the training dataset. Check the path for the testing dataset under section 'LOAD DATASETS'")
	
	test_X = test_data.iloc[:, :-1]
	test_Y = encoder.transform(test_data.iloc[:, -1])
	
	# Making prediction by take mode of predictions 
	# made by all the classifiers
	svm_preds = final_svm_model.predict(test_X)
	nb_preds = final_nb_model.predict(test_X)
	rf_preds = final_rf_model.predict(test_X)
	
	final_preds = [mode([i,j,k])[0] for i,j,
				k in zip(svm_preds, nb_preds, rf_preds)]
	
	st.info("Accuracy by Combined Model")
	st.success(f"On Test Dataset\
		  : {accuracy_score(test_Y, final_preds)*100}")
	
	cf_matrix = confusion_matrix(test_Y, final_preds)
	combined_plot = plt.figure(figsize=(12,8))
	sns.heatmap(cf_matrix, annot = True)
	plt.title("Confusion Matrix for Combined Model on Test Dataset")
	st.pyplot(combined_plot)



# ====================================================
# ROW 4 - Model Testing
#=====================================================
st.divider()
st.markdown('''<div style="font-size:45px; font-weight: bold; padding-bottom: 20px;"> 
			Model Predictions 
			</div>''',unsafe_allow_html=True)
st.write()
symptoms = X.columns.values

# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
	symptom = " ".join([i.capitalize() for i in value.split("_")])
	symptom_index[symptom] = index

data_dict = {
	"symptom_index":symptom_index,
	"predictions_classes":encoder.classes_
}

# Defining the Function
# Input: string containing symptoms separated by commas
# Output: Generated predictions by models
def predictDisease(symptoms):
	symptoms = symptoms.split(",")
	
	# creating input data for the models
	input_data = [0] * len(data_dict["symptom_index"])
	for symptom in symptoms:
		index = data_dict["symptom_index"][symptom]
		input_data[index] = 1
		
	# reshaping the input data and converting it
	# into suitable format for model predictions
	input_data = np.array(input_data).reshape(1,-1)
	
	# generating individual outputs
	rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
	nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
	svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
	
	# making final prediction by taking mode of all predictions
	final_prediction = np.unique([rf_prediction, nb_prediction, svm_prediction])[0]
	predictions = {
		"rf_model_prediction": rf_prediction,
		"naive_bayes_prediction": nb_prediction,
		"svm_model_prediction": svm_prediction,
		"final_prediction":final_prediction
	}
	return predictions

# Allowed input data
row4_col1, row4_col2 = st.columns([1,1],gap="large")

with row4_col2 :
	st.write()
	st.info("The Model is trained for following Range of Symptoms")
	symptoms_list = list(symptoms)
	symptoms_list.sort()
	symtoms_list_regenerated = []
	for __word__ in symptoms_list :
		if "_" in __word__ :
			regenerated_symtom = " ".join(__word__.split("_"))
			symtoms_list_regenerated.append(regenerated_symtom.title())
		else :
			symtoms_list_regenerated.append(__word__.title())
	row4_sub1_col1,row4_sub1_col2 = st.columns(2)
	row4_sub1_col1.write(symtoms_list_regenerated[0:66])
	row4_sub1_col2.write(symtoms_list_regenerated[66:])

with row4_col1 :
	predict_disease_form = st.form("Disease Prediction Form",clear_on_submit=True)
	r1, r2 = predict_disease_form.columns(2)
	symptom_value_range = tuple(["None"] + symtoms_list_regenerated)
	symptom_1 = r1.selectbox("Symptom 1 : ",symptom_value_range, placeholder="Choose an Option")
	symptom_2 = r2.selectbox("Symptom 2 : ",symptom_value_range, placeholder="Choose an Option")
	symptom_3 = r1.selectbox("Symptom 3 : ",symptom_value_range, placeholder="Choose an Option")
	symptom_4 = r2.selectbox("Symptom 4 : ",symptom_value_range, placeholder="Choose an Option")
	symptom_5 = r1.selectbox("Symptom 5 : ",symptom_value_range, placeholder="Choose an Option")
	symptom_6 = r2.selectbox("Symptom 6 : ",symptom_value_range, placeholder="Choose an Option")
	predict_disease_form_submit = predict_disease_form.form_submit_button("Make Predictions")

	if predict_disease_form_submit == True :
		list_entry = [symptom_1,symptom_2,symptom_3,symptom_4,symptom_5,symptom_6]
		symptoms_group_list = []
		for i in list_entry :
			if i != "None" :
				symptoms_group_list.append(i)
		symptoms_group = ",".join(symptoms_group_list)
		usr_entry_dict = {"Symptom Number":["Symptom %d" % j for j in range(1,7)], "Data":list_entry}
		usr_entry_symptoms_df = pd.DataFrame(usr_entry_dict,index=[n for n in range(1,7)])
		st.dataframe(usr_entry_symptoms_df,use_container_width=True,hide_index=True)
		try:
			st.success("Model Predictions") 
			st.write(predictDisease(symptoms_group))
		except Exception as e :
			st.error("There is some error in the symptoms combination.")
	
	st.divider()
	st.warning("WARNING - This model is only for learning and testing purpose and should not be used for giving any actual diagnosis")

st.divider()
st.info(f"Session State ID - {uuid.uuid4()}")

