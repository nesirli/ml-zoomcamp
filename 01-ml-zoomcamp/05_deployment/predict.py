import pandas as pd
import pickle

with open('model.bin', 'rb') as file:
    pretrained_model = pickle.load(file)


customer = {
    'customerid': '8879-zkjof',
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'no',
    'dependents': 'no',
    'tenure': 41,
    'phoneservice': 'yes',
    'multiplelines': 'no',
    'internetservice': 'dsl',
    'onlinesecurity': 'yes',
    'onlinebackup': 'no',
    'deviceprotection': 'yes',
    'techsupport': 'yes',
    'streamingtv': 'yes',
    'streamingmovies': 'yes',
    'contract': 'one_year',
    'paperlessbilling': 'yes',
    'paymentmethod': 'bank_transfer_(automatic)',
    'monthlycharges': 79.85,
    'totalcharges': 3320.75
}

numerical_columns = ['tenure', 'monthlycharges', 'totalcharges']
categorical_columns = [
    'gender', 'seniorcitizen', 'partner', 'dependents', 'phoneservice',
    'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
    'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
    'contract', 'paperlessbilling', 'paymentmethod'
]


def prepare_data(customer_dict):
    df_small = pd.DataFrame([customer_dict])
    df_small['seniorcitizen'] = df_small['seniorcitizen'].astype('str')

    df_small_one_hot_encoded = pd.get_dummies(df_small[categorical_columns], dtype=int)
    df_final = pd.concat([df_small[numerical_columns], df_small_one_hot_encoded], axis=1)
    df_final = df_final.reindex(columns=pretrained_model.feature_names_in_, fill_value=0)

    return df_final

customer_prepared = prepare_data(customer)
churn_probability = pretrained_model.predict_proba(customer_prepared)[:, 1]
print(churn_probability)