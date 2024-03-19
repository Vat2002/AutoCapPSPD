from pickle import load
import cv2
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from ultralytics import YOLO
import openai
from nltk.translate.bleu_score import corpus_bleu
from sentence_transformers import SentenceTransformer
import numpy as np
import constants
from evaluate_model import load_set, load_photo_features, load_clean_descriptions, create_tokenizer, max_length


def extract_features(filename):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature


def get_obj(image_path):
    image = cv2.imread(image_path)
    model = YOLO('model/yolomods/yolov8x-seg.pt')
    results = model.predict(image)
    result = results[0]
    classes = []
    for box in result.boxes:
        class_id = box.cls[0].item()
        classes.append(result.names[class_id])
    return classes


def generate_caption_gpt(vgmodel, tokenizer_m, image_path):
    tokenizer = load(open(tokenizer_m, 'rb'))
    max_length = 34
    model = load_model(vgmodel)
    print("VGG16 Loaded!")
    photo = extract_features(image_path)
    print("Features:")
    image_features = generate_desc(model, tokenizer, photo, max_length)
    print(image_features)
    objects_detected = get_obj(image_path)
    print("Objects:")
    print(objects_detected)
    prompt = (
        f"Write a creative caption for an image with the following features: {image_features} and containing these "
        f"objects: {objects_detected}."
        f"Correct all the grammar mistakes in it and it should produce a caption to an image related to its image "
        f"features and objects.The caption should be meaning.")
    print(prompt)
    openai.api_key = constants.OPENAI_API_KEY
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text


def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    model2 = SentenceTransformer('paraphrase-distilroberta-base-v2')
    total_cosine_similarity = 0
    num_items = 0
    for key, desc_list in descriptions.items():
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
        if references:
            sentence1 = ' '.join(references[0])
            sentence2 = yhat
            print('Actual:')
            print(sentence1)
            print("Predicted:")
            print(sentence2)
            embedding1 = model2.encode(sentence1)
            embedding2 = model2.encode(sentence2)
            cosine_similarity = np.dot(embedding1, embedding2) / (
                    np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            print(f"Cosine Similarity between actual and predicted for {key}: {cosine_similarity}")
            total_cosine_similarity += cosine_similarity
            num_items += 1
    if num_items > 0:
        average_cosine_similarity = total_cosine_similarity / num_items
        print(f"Average Cosine Similarity: {average_cosine_similarity}")
    else:
        print("No items to calculate cosine similarity.")
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# load training dataset (6K)
filename = 'flickr8ktext/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('flickr8ktext/descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('model/features.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# prepare test set
# load test set
filename = 'flickr8ktext/Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('flickr8ktext/descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('model/features.pkl', test)
print('Photos: test=%d' % len(test_features))

# load the model
filename = 'model/kerasmods/model.h5'
model = load_model(filename)
# evaluate model
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

# cap = generate_caption_gpt('model/kerasmods/model.h5', 'model/tokenizer.pkl', 'test_images/dog.jpg')
# print(cap)
