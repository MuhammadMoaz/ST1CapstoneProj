from tkinter import *
import pandas as pd
import pickle


class myGUI:
    # Create a function which is called when predict button is clicked
    def predict_continent(self):
        # Get the input features from the entry boxes
        population = float(self.populationEntry.get())
        imfGDP = float(self.imfEntry.get())
        unGDP = float(self.unEntry.get())
        gdpPerCapita = float(self.gdpPerCapEntry.get())

        # Create a dataframe with the input features
        data = pd.DataFrame({'Population': [population],
                             'IMF_GDP': [imfGDP],
                             'UN_GDP': [unGDP],
                             'GDP_per_capita': [gdpPerCapita]})

        # Use the model to predict the continent
        prediction = self.model.predict(data)[0]

        # Display the predicted continent
        self.resultLabel.config(text='Predicted Continent: {}'.format(self.continent_dict[prediction]), bg='Lightblue')

    def __init__(self):
        # Load the best performing model
        with open('best_model.pkl', 'rb') as f:
            self.model = pickle.load(f)

        # Create a dictionary of the continent labels
        self.continent_dict = {
            0: 'Africa',
            1: 'Asia',
            2: 'Europe',
            3: 'North America',
            4: 'Oceania',
            5: 'South America'
        }

        # Create the GUI
        self.root = Tk()
        self.root.title('Continent Predictor')

        # Create label and entry for population input
        self.populationLabel = Label(self.root, text='Population')
        self.populationEntry = Entry(self.root)

        # Pack population label and entry
        self.populationLabel.grid(row=0)
        self.populationEntry.grid(row=0, column=1)

        # Create label and entry for IMF GDP input
        self.imfLabel = Label(self.root, text='IMF GDP')
        self.imfEntry = Entry(self.root)

        # Pack IMF GDP label and entry
        self.imfLabel.grid(row=1)
        self.imfEntry.grid(row=1, column=1)

        # Create label and entry for UN GDP input
        self.unLabel = Label(self.root, text='UN GDP')
        self.unEntry = Entry(self.root)

        # Pack UN GDP label and entry
        self.unLabel.grid(row=2)
        self.unEntry.grid(row=2, column=1)

        # Create label and entry for GDP per capita
        self.gdpPerCapLabel = Label(self.root, text='GDP per capita')
        self.gdpPerCapEntry = Entry(self.root)

        self.gdpPerCapLabel.grid(row=3)
        self.gdpPerCapEntry.grid(row=3, column=1)

        # Create predict and quit button
        self.predButton = Button(self.root, text='Predict', command=self.predict_continent)
        self.quitButton = Button(self.root, text='Quit', command=self.root.destroy)

        # Packing predict and quit button
        self.predButton.grid(row=4, column=0, sticky=E)
        self.quitButton.grid(row=4, column=1, sticky=W)

        # Create and pack a label to display the predicted continent
        self.resultLabel = Label(self.root, text='')
        self.resultLabel.grid(row=5, column=0, columnspan=2)

        # Start the GUI
        self.root.mainloop()


my_GUI = myGUI()

