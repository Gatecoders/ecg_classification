import transform_images
import training
import prediction

def main():
    """_summary_
    """    
    transform = 1                          # training = 0, prediction = 1, transformation = 2
    model_type = 1                               # 0 = rf, 1 = dnn
    model_name = 'new_dnn3'                 


    if transform == 0 :
        training.train(model_name, model_type)
    elif transform == 1:
        print('Transforming images')
        transformed_file = transform_images.graph_extraction(transform, model_name, model_type)
        print('Images transformed')
        prediction.predict(transformed_file, model_name, model_type)
    else:
        transformed_file = transform_images.graph_extraction(transform, model_name, model_type)

if __name__ == '__main__':
    main()