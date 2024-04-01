import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from os import listdir
from os.path import join

def load_images_and_labels(folder):
    images = []
    labels = []
    for filename in listdir(folder):
        img = imread(join(folder, filename))
        if img is not None:
            images.append(img)
            # Label assignment based on filename
            if 'cat' in filename.lower():
                labels.append(0)  # 0 for cats
            elif 'dog' in filename.lower():
                labels.append(1)  # 1 for dogs
    return images, labels

def create_features(img):
    resized_img = resize(img, (128, 64))  # Resize image to uniform size
    # Extract HOG features
    features = hog(resized_img, orientations=8, pixels_per_cell=(16, 16),
                   cells_per_block=(1, 1), channel_axis=-1)
    return features

train_folder = r"C:\Users\nares\PycharmProjects\SVM\.venv\dogs-vs-cats\train\train" #Here add the path for your trainning dataset folder
test_folder = r"C:\Users\nares\PycharmProjects\SVM\.venv\dogs-vs-cats\test1\test1" #Here add the path for your test dataset folder

train_images, train_labels = load_images_and_labels(train_folder)
test_images, test_labels = load_images_and_labels(test_folder)


train_features = np.array([create_features(img) for img in train_images])
test_features = np.array([create_features(img) for img in test_images])


scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

clf = SVC(kernel='linear', C=1.0)
clf.fit(train_features_scaled, train_labels)

predictions = clf.predict(test_features_scaled)
accuracy = accuracy_score(test_labels, predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

#Use a dataset containing the images of cat and dogs with file name specified as cat_01.png,dog_01.png
