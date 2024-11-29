import os
import joblib
import pandas as pd
from config import MDLS_DIR
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

def predict(transformed_file, model_name, model_type):
    df=transformed_file
    mdls_dir = MDLS_DIR
    x_test=df.drop(columns=['target'])
    y_test= df.target
    
    if model_type==0:
        model_path = os.path.join(mdls_dir, f'rf_{model_name}.pkl')
        model = joblib.load(model_path)
    else:
        model_path = os.path.join(mdls_dir, f'dnn_{model_name}.h5')
        model = load_model(model_path)

    
    # y_pred = (model.predict(x_test) > 0.5).astype(int)     #for binary classification
    # import pdb
    # pdb.set_trace()
    y_pred = model.predict(x_test)
    if model_type == 1:
        y_pred = y_pred.argmax(axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    y_test = pd.DataFrame(df['target'], columns=['target'])
    y_pred_df = pd.DataFrame(y_pred, columns=['pred'])
    results = pd.concat([y_test.reset_index(drop=True), y_pred_df.reset_index(drop=True)], axis=1)
    print(results)
