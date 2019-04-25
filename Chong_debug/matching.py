import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import pandas as pd
from ast import literal_eval
import os, os.path

img_input_path = 'input_img/input.jpg'
embedding_path = 'embedding_clean.csv'
merged_path = 'clean.csv'
matching_output_path = 'output.txt'

def cos_d(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return 1-v1.dot(v2.T)/np.linalg.norm(v1)/np.linalg.norm(v2)

def matching(img_input_path, embedding_path, merged_path, matching_output_path):
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    img = image.load_img(img_input_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    emb_input = model.predict(img_data)[0]
    
    embedding = pd.read_csv(embedding_path)
    merged = pd.read_csv(merged_path)

    emb_matrix = np.array([literal_eval(emb) for emb in embedding.emb])

    scores = []
    for i in range(len(emb_matrix)):
        scores.append(cos_d(emb_input, list(emb_matrix[i])))
    
    
    
    ranks = [[index, merged.photo_url[index], merged.post_url[index], 
              merged.meta[index], score] for index, score in enumerate(scores, 0)]
    ranks.sort(key=lambda x:x[-1])
    
    return str([ranks[0],ranks[1]])

def main():
    output = matching(img_input_path, embedding_path, merged_path, matching_output_path)
    with open(matching_output_path, 'w') as file:
        file.write(output)
    print("===Done! Matching output file is saved.===")
    
if __name__ == "__main__":
    main()
