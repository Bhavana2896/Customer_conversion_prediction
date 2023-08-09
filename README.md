Customer Conversion Prediction for Insurance

Problem Statement
In the context of a new-age insurance company, the goal is to identify potential customers who are likely to subscribe to the insurance. This machine learning project aims to predict whether a client will subscribe to the insurance based on historical marketing data.

Data
The historical sales data is available in the data directory of this repository.

Features:

age (numeric)
job: type of job
marital: marital status
educational_qual: education status
call_type: contact communication type
day: last contact day of the month (numeric)
mon: last contact month of year
dur: last contact duration, in seconds (numeric)
num_calls: number of contacts performed during this campaign and for this client
prev_outcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

Output variable (desired target):
y: Has the client subscribed to the insurance?

Steps Followed in Model Development
Import Libraries: Import necessary libraries for data manipulation, visualization, and model building.

Data Cleaning: Handle missing values, duplicates, and outliers. Correct data types.

Exploratory Data Analysis (EDA): Visualize data to understand distributions and relationships w.r.t. target.

Data Preprocessing:
Remove unnecessary columns.
Encode categorical features.

Class Balance: Check and balance the target classes using techniques like SMOTEENN.

Data Scaling: Scale numerical features to a common scale.

Model Building: Train and evaluate various models: Logistic Regression, KNN, Decision Tree, Voting Classifier, Random Forest, XGBoost.

Model Evaluation: Compare model performance using metrics like accuracy and F1-score.

Feature Importance: Determine XGBoost's(best model here) feature importances.

Final Model: Select the best-performing model and retrain on the entire training data.

Model Persistence: Save the trained model using pickling.

App Development: Create a user-friendly Streamlit app for deployment.

Map user inputs to encoded values.
Validate input and scale it.
Make predictions and display results.

Deployment: Deploy the Streamlit app on platforms Streamlit Sharing.

The above steps ensure the development of a machine learning model for customer conversion prediction, offering insights into potential subscribers for the insurance company.




