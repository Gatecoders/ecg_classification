# import pickle
# import pandas as pd
# from config import MDLS_DIR
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.ensemble import RandomForestClassifier

# def train(transformed_file, model_name,model_type):
#     mdls_dir = MDLS_DIR
#     df = transformed_file
#     x_train=df.drop(columns=['target'])
#     y_train= pd.DataFrame(df.target,columns=['target'])

#     if model_type==0:
#         model=RandomForestClassifier(random_state=67,
#                                   n_estimators=50,
#                                   verbose=3,
#                                   n_jobs=-1)
#         model.fit(x_train,y_train)

#     else:
#         model=create_dnn_model(x_train.shape)
#         early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#         model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

#     if model_type==0:
#         model_path = f'{mdls_dir}/rf_{model_name}.pkl'
#     else:
#         model_path = f'{mdls_dir}/dnn_{model_name}.h5'
#     with open(model_path, 'wb') as f:
#         pickle.dump(model, f)

# def create_dnn_model(x, 
#                      l1_out=50, l1_drop=0.5, l1_act='relu',
#                      l2_out=25, l2_drop=0.3, l2_act='relu',
#                      l3_out=10, l3_drop=0.2, l3_act='relu',
#                      l4_out=1, l4_act='sigmoid'):
    
#     model = Sequential()
    
#     # Layer 1
#     model.add(Dense(l1_out, activation=l1_act, input_shape=(x[1],)))
#     model.add(Dropout(l1_drop))
    
#     # Layer 2
#     model.add(Dense(l2_out, activation=l2_act))
#     model.add(Dropout(l2_drop))
    
#     # Layer 3
#     model.add(Dense(l3_out, activation=l3_act))
#     model.add(Dropout(l3_drop))
    
#     # Layer 4
#     model.add(Dense(l4_out, activation=l4_act))

#     model.compile(optimizer=Adam(),
#               loss='binary_crossentropy',  #multi: sparse_categorical_crossentropy, binary: binary_crossentropy 
#               metrics=['accuracy'])
    
#     return model     


import pickle
import pandas as pd
from config import MDLS_DIR, data_file_dir
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier

def train(model_name, model_type):
    data_file = data_file_dir
    mdls_dir = MDLS_DIR
    df = pd.read_csv(f'{data_file}/pca_final.csv')
    x_train = df.drop(columns=['target'])
    y_train = df['target']
    
    # Calculate the number of unique classes in the target variable
    num_classes = y_train.nunique()

    if model_type == 0:
        # Train a RandomForestClassifier for multiclass classification
        model = RandomForestClassifier(random_state=67, n_estimators=20, verbose=3, n_jobs=-1)
        model.fit(x_train, y_train)
    else:
        # Train a DNN model for multiclass classification
        model = create_dnn_model(x_train.shape, num_classes)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

    # Save the trained model
    if model_type == 0:
        model_path = f'{mdls_dir}/rf_{model_name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        model_path = f'{mdls_dir}/dnn_{model_name}.h5'
        model.save(model_path)

def create_dnn_model(input_shape, num_classes,
                     l1_out=50, l1_drop=0.5, l1_act='relu',
                     l2_out=25, l2_drop=0.3, l2_act='relu',
                     l3_out=10, l3_drop=0.2, l3_act='relu'):
    
    model = Sequential()
    
    # Layer 1
    model.add(Dense(l1_out, activation=l1_act, input_shape=(input_shape[1],)))
    model.add(Dropout(l1_drop))
    
    # Layer 2
    model.add(Dense(l2_out, activation=l2_act))
    model.add(Dropout(l2_drop))
    
    # Layer 3
    model.add(Dense(l3_out, activation=l3_act))
    model.add(Dropout(l3_drop))
    
    # Output Layer for multiclass classification
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model with the appropriate loss function for multiclass classification
    model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',  # sparse_categorical_crossentropy, binary_crossentropy
                  metrics=['accuracy'])
    
    return model
