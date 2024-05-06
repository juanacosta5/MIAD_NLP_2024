import joblib
import os
import sys
import pandas as pd

clf = joblib.load('stacking_model.pkl')

def predict(data):
    try:
        # cargamos
        input_data = pd.DataFrame([data])

        # predecimos
        prediction = clf.predict(input_data)

        # resultado
        return prediction[0],200
    except Exception as e:
        return "Error ",500

if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add data')
        
    else:

        data = sys.argv[1]

        p1 = predict(data)
        
        print(data)
        print('Price of Car: ', p1)