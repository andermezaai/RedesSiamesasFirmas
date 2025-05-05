import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda,BatchNormalization, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import numpy as np
import random
import numpy as np
import cv2
import os

def processImage(input_path, target_size):    
    # Cargar la imagen
    img = cv2.imread(input_path)

    # Convertir la imagen a escala de grises para análisis del fondo
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar el algoritmo de Otsu para obtener el umbral óptimo
    _, umbral = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)        

    # Calcular la relación de aspecto
    aspect_ratio = img.shape[1] / img.shape[0]
    aspect_ratio_target = target_size[1] / target_size[0]
    
    # Determinar el nuevo tamaño manteniendo la relación de aspecto
    if aspect_ratio_target > aspect_ratio:
        new_height = img.shape[0]
        new_width = int(aspect_ratio_target * img.shape[0])
    else:
        new_height = int(img.shape[1] / aspect_ratio_target)
        new_width = img.shape[1]
    
    # Contar los píxeles blancos y negros para determinar el color predominante
    num_white = np.sum(umbral == 255)
    num_black = np.sum(umbral == 0)

    # Si hay más píxeles negros que blancos, se considera que el fondo original es oscuro
    if num_black > num_white:
        imagen_invertida = umbral
    else:
        # Invertir la imagen
        imagen_invertida = 255 - umbral

    # Crear una nueva imagen con fondo blanco
    new_img = np.ones((new_height, new_width), dtype=np.uint8) * 0
    
    # Calcular las coordenadas para centrar la imagen redimensionada
    top_left_x = (new_width - img.shape[1]) // 2
    top_left_y = (new_height - img.shape[0]) // 2
    
    # Pegar la imagen redimensionada en la nueva imagen
    new_img[top_left_y:top_left_y + img.shape[0], top_left_x:top_left_x + img.shape[1]] = imagen_invertida
    
    # Redimensionar la imagen al tamaño objetivo
    new_img = cv2.resize(new_img, (target_size[1],target_size[0]))        
    
    return new_img

def argument_image(image,iguales):
    if not iguales:
        # Rotación
        angle = random.uniform(-15, 15)  # Rotar en un rango de -15 a 15 grados
        M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1)
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Desplazamiento
    tx = random.uniform(-10, 10)  # Desplazar en un rango de -10 a 10 píxeles en x
    ty = random.uniform(-10, 10)  # Desplazar en un rango de -10 a 10 píxeles en y
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Recorte
    start_x = max(0, int((translated.shape[1] - image.shape[1]) / 2))
    start_y = max(0, int((translated.shape[0] - image.shape[0]) / 2))
    cropped = translated[start_y:start_y + image.shape[0], start_x:start_x + image.shape[1]]

    return cropped

def extraerImagenesNegativas(folder,input_shape):
    dataset=[]
    for subfolder in os.listdir(folder):    
        if "forg" not in subfolder:
            print(subfolder)
            rutaSubfolderA=folder+"\\"+subfolder #obtenemos la ruta de la carpeta firmas originales
            rutaSubfolderB=folder+"\\"+subfolder+"_forg"#obtenemos la ruta de la carpeta de firmas falsas
            for imagen in os.listdir(rutaSubfolderA):
                imagenA=processImage(rutaSubfolderA+"\\"+imagen,input_shape)                
                dataset.extend([([imagenA,processImage(rutaSubfolderB+"\\"+imagenFalsa,input_shape)],0) for imagenFalsa in os.listdir(rutaSubfolderB)])            
    return dataset

def extraerImagenesPositivas(folder,input_shape):
    dataset=[]
    for subfolder in os.listdir(folder):    
        if "forg" not in subfolder:
            print(subfolder)
            rutaSubfolderA=folder+"\\"+subfolder #obtenemos la ruta de la carpeta firmas originales
            for imagen in os.listdir(rutaSubfolderA):
                imagenA=processImage(rutaSubfolderA+"\\"+imagen,input_shape)                
                dataset.extend([([imagenA,processImage(rutaSubfolderA+"\\"+imagenReal,input_shape)],1) for imagenReal in os.listdir(rutaSubfolderA) if imagenReal!=imagen])           
    return dataset

def extraerImagenesOtros(folder,input_shape):
    dataset=[]
    for subfolder in os.listdir(folder):    
        if "forg" not in subfolder:
            print(subfolder)
            rutaSubfolderA=folder+"\\"+subfolder #obtenemos la ruta de la carpeta firmas originales
            for imagen in os.listdir(rutaSubfolderA):
                imagenA=processImage(rutaSubfolderA+"\\"+imagen,input_shape)                
                dataset.extend([([imagenA,processImage(folder+"\\"+subfolderNegativas+"\\"+os.listdir(folder+"\\"+subfolderNegativas)[0],input_shape)],0) for subfolderNegativas in os.listdir(folder) if subfolder not in subfolderNegativas])                     
    return dataset

def extractorDataset(folder, input_shape, batch_size):
    dataset = []
    intra_clase_real_count = 0
    inter_clase_real_falsificada_count = 0
    inter_clase_diferentes_personas_count = 0

    intra_clase_real_limit = int(batch_size * 0.5)
    inter_clase_real_falsificada_limit = int(batch_size * 0)
    inter_clase_diferentes_personas_limit = int(batch_size * 0.5)

    while True:
        subfolder=random.choice([i for i in os.listdir(folder) if "forg" not in i])
        rutaSubfolderA = os.path.join(folder, subfolder)
        rutaSubfolderB = os.path.join(folder, f"{subfolder}_forg")

        imagen = random.choice(os.listdir(rutaSubfolderA))
        imagenA = processImage(os.path.join(rutaSubfolderA, imagen), input_shape)   

        if intra_clase_real_count < intra_clase_real_limit:                        
            imagenReal = random.choice([i for i in os.listdir(rutaSubfolderA) if i != imagen])
            imagenB = processImage(os.path.join(rutaSubfolderA, imagenReal), input_shape)
            # Decidir aleatoriamente si aplicar augmentación
            aplicar_aug_A = random.choice([True, False])
            aplicar_aug_B = random.choice([True, False])
            # Aplicar augmentación si la condición es True, sino usar la imagen original
            imagenA_augmented = argument_image(imagenA, True) if aplicar_aug_A else imagenA
            imagenB_augmented = argument_image(imagenB, True) if aplicar_aug_B else imagenB
            dataset.append([[imagenA_augmented,imagenB_augmented], 1])
            intra_clase_real_count += 1                        
        elif inter_clase_real_falsificada_count < inter_clase_real_falsificada_limit:
            # Inter-clase Real-Falsificada                   
            imagenFalsa = random.choice(os.listdir(rutaSubfolderB))
            imagenB = processImage(os.path.join(rutaSubfolderB, imagenFalsa), input_shape)
            # Decidir aleatoriamente si aplicar augmentación
            aplicar_aug_A = random.choice([True, False])
            aplicar_aug_B = random.choice([True, False])
            # Aplicar augmentación si la condición es True, sino usar la imagen original
            imagenA_augmented = argument_image(imagenA, False) if aplicar_aug_A else imagenA
            imagenB_augmented = argument_image(imagenB, False) if aplicar_aug_B else imagenB
            dataset.append([[imagenA_augmented,imagenB_augmented], 0])
            inter_clase_real_falsificada_count +=1                      
        elif inter_clase_diferentes_personas_count < inter_clase_diferentes_personas_limit:
            # Inter-clase Diferentes Personas                    
            subfolderNegativas = random.choice([i for i in os.listdir(folder) if subfolder not in i])
            imagenB = processImage(os.path.join(folder, subfolderNegativas, random.choice(os.listdir(os.path.join(folder, subfolderNegativas)))), input_shape)
            # Decidir aleatoriamente si aplicar augmentación
            aplicar_aug_A = random.choice([True, False])
            aplicar_aug_B = random.choice([True, False])
            # Aplicar augmentación si la condición es True, sino usar la imagen original
            imagenA_augmented = argument_image(imagenA, False) if aplicar_aug_A else imagenA
            imagenB_augmented = argument_image(imagenB, False) if aplicar_aug_B else imagenB
            dataset.append([[imagenA_augmented,imagenB_augmented], 0])
            inter_clase_diferentes_personas_count += 1                       
            
        # Reset counts if batch is filled
        if len(dataset) >= batch_size:
            intra_clase_real_count = 0
            inter_clase_real_falsificada_count = 0
            inter_clase_diferentes_personas_count = 0
            yield preparar_lote(dataset)
            dataset = []

def preparar_lote(dataset):

    x1 = np.array([par[0][0] for par in dataset],dtype=np.float32)
    x2 = np.array([par[0][1] for par in dataset],dtype=np.float32)
    y = np.array([par[1] for par in dataset],dtype=np.float32)
    return [x1,x2], y

def build_siamese_model(input_shape):
    # Definición de la arquitectura de la subred
    input = Input(shape=input_shape)
    x = Conv2D(48, (6, 12),strides=(4, 4), padding='valid', activation='relu')(input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = Conv2D(128, (5, 5),strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D( (4,4) ,strides=(2, 2), padding='valid')(x)
    x = Conv2D(192, (4, 4),strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (4, 4),strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (4, 4),strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(260, (4, 4),strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D( (3,3) ,strides=(2, 2), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)    
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    model = Model(inputs=input, outputs=x)
    return model

def build_siamese_model1(input_shape):
    # Definición de la arquitectura de la subred
    input = Input(shape=input_shape)
    x = Conv2D(120, (11, 11),strides=(3, 3), padding='valid', activation='relu')(input)#120
    x = MaxPooling2D(strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(300, (5, 5),strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(400, (3, 3),strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(400, (3, 3),strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(280, (3, 3),strides=(1, 1), padding='same', activation='relu')(x)#280
    x = MaxPooling2D(strides=(2, 2))(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(2420, activation='relu')(x)#2420
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(2420, activation='relu')(x) #2420
    x = Dropout(0.4)(x) #0.4
    x = BatchNormalization()(x)
    model = Model(inputs=input, outputs=x)
    return model

def euclidean_distance(vectors):
    x, y = vectors
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(sum_square)

def build_complete_siamese_network(input_shape):
    # Crear la subred
    base_network = build_siamese_model(input_shape)
    
    # Entradas para las dos imágenes
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    # Obtener la representación de las imágenes usando la misma red
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Concatenar las dos salidas
    distance = Lambda(euclidean_distance)([processed_a, processed_b])
    
    # Añadir una capa densa después de la concatenación
    #output = Dense(1, activation='sigmoid')(distance)  # Para obtener una salida final (0 o 1 para clasificación)
    
    # Modelo final
    model = Model(inputs=[input_a, input_b], outputs=distance)
    return model
 
def build_complete_siamese_network2(input_shape):
    # Crear la subred
    input = Input(shape=input_shape)
    x = Conv2D(120, (3, 3),strides=(1, 1), padding='valid', activation='relu',kernel_regularizer=regularizers.l2(0.01))(input)#120
    x = MaxPooling2D(strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(300, (4, 4),strides=(2, 2), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(400, (3, 3),strides=(1, 1), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Conv2D(400, (3, 3),strides=(1, 1), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Conv2D(280, (3, 3),strides=(1, 1), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)#280
    x = MaxPooling2D(strides=(2, 2))(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(2420, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)#2420
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(2420, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x) #2420
    x = Dropout(0.4)(x) #0.4
    x = BatchNormalization()(x)   
    x = Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)  # Puedes ajustar el tamaño de esta capa según sea necesario
    x = Dense(1, activation='sigmoid')(x)  # Para obtener una salida final (0 o 1 para clasificación)
    
    # Modelo final
    model = Model(inputs=input, outputs=x)
    return model

def contrastive_loss(y_true, y_pred):
    margin = 1
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

