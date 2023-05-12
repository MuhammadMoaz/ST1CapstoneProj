import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle

if __name__ == '__main__':
    # Load dataset from csv file
    data = pd.read_csv('countries.csv')

    # Check first and last 5 rows of data to ensure it's loaded correctly
    print(data.head())
    print(data.tail())

    # Understanding shape of dataset (rows and columns)
    print(f'\nDataset Shape: {data.shape}')

    # Summary stats of the data (count, mean, std, min, etc)
    print(data.describe())

    # Drop any rows with missing values
    data.dropna(inplace=True)

    # General EDA

    # Distribution of GDP per capita across all countries
    plt.hist(data['GDP_per_capita'], color='purple')
    plt.title('Distribution of GDP per capita')
    plt.xlabel('GDP per capita')
    plt.ylabel('Count')
    plt.show()

    # Scatter plot of population vs. GDP
    plt.scatter(data['Population'], data['IMF_GDP'], c='purple')
    plt.title('Population vs. IMF GDP')
    plt.xlabel('Population')
    plt.ylabel('IMF GDP')
    plt.show()

    # Correlation matrix heatmap
    numeric_cols = ['Rank', 'ID', 'Population', 'IMF_GDP', 'UN_GDP', 'GDP_per_capita']
    numeric_data = data[numeric_cols]

    corr_matrix = numeric_data.corr()
    plt.title('Correlation Matrix Heatmap')
    plt.imshow(corr_matrix, cmap='PRGn', interpolation='nearest')
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.colorbar()
    plt.show()

    # Q1 Visualisation: What is the distribution of GDP per capita across different continents?

    # Create a box plot of GDP per capita by continent
    data.boxplot(column='GDP_per_capita', by='Continent', vert=False, grid=False, color='purple')
    plt.title('Distribution of GDP per capita by Continent')
    plt.xlabel('GDP per capita')
    plt.ylabel('Continent')
    plt.show()

    # Q2 Visualisation: Is there a relationship between a country's population and its GDP per capita?

    # Create a scatter plot of population vs. GDP per capita
    plt.scatter(data['Population'], data['GDP_per_capita'], c='purple')
    plt.title('Relationship between Population and GDP per capita')
    plt.xlabel('Population')
    plt.ylabel('GDP per capita')
    plt.show()

    # Q3 Visualisation: Are there any notable differences in the GDP estimates reported by the IMF and UN for different countries?

    # Calculate the difference between IMF and UN GDP estimates
    data['GDP_diff'] = data['IMF_GDP'] - data['UN_GDP']

    # Create a scatter plot of the IMF vs UN GDP estimates
    plt.scatter(data['IMF_GDP'], data['UN_GDP'], s=50, c=data['GDP_diff'], cmap='PRGn')
    plt.colorbar()
    plt.title('IMF vs UN GDP Estimates')
    plt.xlabel('IMF GDP Estimate')
    plt.ylabel('UN GDP Estimate')
    plt.show()

    # Q4 Visualisation: What is the overall distribution of population across different continents?

    # Create a bar chart of population by continent
    data.groupby('Continent')['Population'].sum().plot(kind='bar', color='purple')
    plt.title('Total Population by Continent')
    plt.xlabel('Continent')
    plt.ylabel('Population')
    plt.show()

    # Q5 Visualisation: Is there a correlation between a country's rank (in terms of population, GDP, or other measures) and its continent?

    # Create a scatter plot of rank vs. continent
    plt.scatter(data['Rank'], data['Continent'], c='purple')
    plt.title('Relationship between Rank and Continent')
    plt.xlabel('Rank')
    plt.ylabel('Continent')
    plt.show()

    # PDA

    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['Country'] = label_encoder.fit_transform(data['Country'])
    data['Continent'] = label_encoder.fit_transform(data['Continent'])

    # Split data into features and target
    X = data.drop(['Rank', 'ID', 'GDP_diff', 'Country', 'Continent'], axis=1)
    y = data['Continent']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    nb = GaussianNB()
    svc = SVC()
    gbc = GradientBoostingClassifier()
    rfc = RandomForestClassifier()
    dt = DecisionTreeClassifier()
    xgb_model = xgb.XGBClassifier()

    # Fit models to training data
    nb.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    gbc.fit(X_train, y_train)
    rfc.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # Make predictions on test data
    nb_preds = nb.predict(X_test)
    svc_preds = svc.predict(X_test)
    gbc_preds = gbc.predict(X_test)
    rfc_preds = rfc.predict(X_test)
    dt_preds = dt.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)

    # Calculate accuracy scores
    nb_acc = accuracy_score(y_test, nb_preds)
    svc_acc = accuracy_score(y_test, svc_preds)
    gbc_acc = accuracy_score(y_test, gbc_preds)
    rfc_acc = accuracy_score(y_test, rfc_preds)
    dt_acc = accuracy_score(y_test, dt_preds)
    xgb_acc = accuracy_score(y_test, xgb_preds)

    # Calculate precision scores
    nb_prec = precision_score(y_test, nb_preds, average='weighted')
    svc_prec = precision_score(y_test, svc_preds, average='weighted')
    gbc_prec = precision_score(y_test, gbc_preds, average='weighted')
    rfc_prec = precision_score(y_test, rfc_preds, average='weighted')
    dt_prec = precision_score(y_test, dt_preds, average='weighted')
    xgb_prec = precision_score(y_test, xgb_preds, average='weighted')

    # Calculate recall scores
    nb_rec = recall_score(y_test, nb_preds, average='weighted')
    svc_rec = recall_score(y_test, svc_preds, average='weighted')
    gbc_rec = recall_score(y_test, gbc_preds, average='weighted')
    rfc_rec = recall_score(y_test, rfc_preds, average='weighted')
    dt_rec = recall_score(y_test, dt_preds, average='weighted')
    xgb_rec = recall_score(y_test, xgb_preds, average='weighted')

    # Calculate f1 scores
    nb_f1 = f1_score(y_test, nb_preds, average='weighted')
    svc_f1 = f1_score(y_test, svc_preds, average='weighted')
    gbc_f1 = f1_score(y_test, gbc_preds, average='weighted')
    rfc_f1 = f1_score(y_test, rfc_preds, average='weighted')
    dt_f1 = f1_score(y_test, dt_preds, average='weighted')
    xgb_f1 = f1_score(y_test, xgb_preds, average='weighted')

    # Model Performance Evaluation

    # Display accuracy scores
    print('===============Model Accuracy Score===============')
    print("Gaussian Naive Bayes accuracy: {:.2f}%".format(nb_acc * 100))
    print("Support Vector Machine accuracy: {:.2f}%".format(svc_acc * 100))
    print("Gradient Boosting Classifier accuracy: {:.2f}%".format(gbc_acc * 100))
    print("Random Forest Classifier accuracy: {:.2f}%".format(rfc_acc * 100))
    print("Decision Tree Classifier accuracy: {:.2f}%".format(dt_acc * 100))
    print("XGBoost Accuracy: {:.2f}%".format(xgb_acc * 100))

    # Display precision scores
    print('\n===============Model Precision Score===============')
    print("Gaussian Naive Bayes precision: {:.2f}%".format(nb_prec * 100))
    print("Support Vector Machine precision: {:.2f}%".format(svc_prec * 100))
    print("Gradient Boosting Classifier precision: {:.2f}%".format(gbc_prec * 100))
    print("Random Forest Classifier precision: {:.2f}%".format(rfc_prec * 100))
    print("Decision Tree Classifier precision: {:.2f}%".format(dt_prec * 100))
    print("XGBoost precision: {:.2f}%".format(xgb_prec * 100))

    # Display recall scores
    print('\n===============Model Recall Score===============')
    print("Gaussian Naive Bayes recall: {:.2f}%".format(nb_rec * 100))
    print("Support Vector Machine recall: {:.2f}%".format(svc_rec * 100))
    print("Gradient Boosting Classifier recall: {:.2f}%".format(gbc_rec * 100))
    print("Random Forest Classifier recall: {:.2f}%".format(rfc_rec * 100))
    print("Decision Tree Classifier recall: {:.2f}%".format(dt_rec * 100))
    print("XGBoost recall: {:.2f}%".format(xgb_prec * 100))

    # Display F1 scores
    print('\n===============Model F1 Score===============')
    print("Gaussian Naive Bayes f1: {:.2f}%".format(nb_f1 * 100))
    print("Support Vector Machine f1: {:.2f}%".format(svc_f1 * 100))
    print("Gradient Boosting Classifier f1: {:.2f}%".format(gbc_f1 * 100))
    print("Random Forest Classifier f1: {:.2f}%".format(rfc_f1 * 100))
    print("Decision Tree Classifier f1: {:.2f}%".format(dt_f1 * 100))
    print("XGBoost f1: {:.2f}%".format(xgb_f1 * 100))

    # Model Performance Evaluation
    best_model = xgb_model
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(y_pred)

    # Classification Report
    print('==========XGBoost Model Classification Report==========')
    print('\n', classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    # Save the data to use in the software tool
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
