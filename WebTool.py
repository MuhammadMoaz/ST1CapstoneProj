from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained machine learning model
model = pickle.load(open('best_model.pkl', 'rb'))


# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')


# Define the prediction page route
@app.route('/predict', methods=['POST'])
def predict():
    # Create a dictionary of the continent labels
    continent_dict = {
        0: 'Africa',
        1: 'Asia',
        2: 'Europe',
        3: 'North America',
        4: 'Oceania',
        5: 'South America'
    }

    # Get the values from the form
    imfGDP = float(request.form['IMF_GDP'])
    unGDP = float(request.form['UN_GDP'])
    population = float(request.form['population'])
    gdpPerCapita = float(request.form['GDP_per_capita'])

    # Create a dataframe with the input features
    data = pd.DataFrame({'Population': [population],
                         'IMF_GDP': [imfGDP],
                         'UN_GDP': [unGDP],
                         'GDP_per_capita': [gdpPerCapita]})

    # Use the trained model to make a prediction, use the prediction as the key in the continent dict to get the value, the continent
    prediction = f'I predict that the continent your country is located in is {continent_dict[model.predict(data)[0]]}'

    # Return the prediction to the user
    return render_template('index.html', prediction=prediction)


# Run the application
if __name__ == '__main__':
    app.run(debug=True)
