# Disaster Response Pipeline Project

### Summary
The following repository is made to create a pipeline for a twit analyzer, which will able a website to detect wheter a twit is a request for help, in this way, authorities will be able to go to a disaster site offering help to whoever is needing it.

The project creates a process of Extract, Transform and Load and saves it in a database accesible with SQL queries, then it creates a pipeline which solves the multiclass problem of defining which kind of help and which kind of disaster is the twit about.

The last part is a website with a visualization of the train dataset and the implementation of the model where you could write your own twit and use the model to test which category is read in it

### Files in the repository
1. app (folder)
1.1. templates (folder)
1.1.1. go.html: Web app of the classification
1.1.2. master.html: main site of the web app
1.2. run.py: Python code which runs the website
2. data (folder)
2.1. disaster_categories.csv: CSV with the categories of the disasters
2.2. disaster_messages.csv: CSV which contains the original message their traslation and the possible categories which it belong to
2.3. messages.db: Created database with the information
2.4. process_data.py: Python ETL which creates the database
3. models (folder)
3.1. classifier.pkl: Resulting model
3.2. train_classifier.pkl: This python models the data and exports the model in pickle
4. readme.md: This file

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
