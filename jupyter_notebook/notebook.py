# %% [markdown]
# ---
# # <div align="center"><font color='pink'> COSC 2673/2793 | Machine Learning</font></div>
# ## <div align="center"> <font size=50 color='pink'>Classify Images of Cancer</font></div>
# ---
#
# # Acknowledgments:
#
# Thank the Machine Learning course Sem 1 2023 instructors at RMIT University for providing me with valuable knowledge, guidance, and support throughout this report. The course lectures, slides, code examples, and tutorials have been instrumental in developing my understanding of various machine-learning concepts and techniques.
#
# # Introduction
#
# The dataset description contains 27x27 RGB images of cells from 99 different patients. The images are histopathology images that show cells at the microscopic level. The goal is to classify these cells based on whether they are cancerous and also to classify them according to their cell type.
#
#
#
# <a id="0"></a>
# # TABLE OF CONTENTS
#
# <br>
#
# ## 1. [Import](#1)
#
# ## 2. [Data Overview](#2)
#
# ## 3. [Data Exploratory Analysis](#3)
#
# ### *3.1. [Correlation between variables](#3.1)*
#
# ### *3.2. [Analyze and Confirm image dimensions](#3.2)*
#
# ### *3.3. [Analyze class distribution](#3.3)*
#
# ### *3.4. [Visualize sample images](#3.4)*
#
# ### *3.5. [Analyze color distributions](#3.5)*
#
# ### *3.6. [Check for duplicated images](#3.6)*
#
# ## 4. [Patient-wise train-test-validation split](#4)
#
# ## 5. [Data Preprocessing](#5)
#
# ## 6. [Evaluation Metrics](#6)
#
# ## 7. [Baseline Model - Traditional Neural Network (not CNN)](#7)
#
# ## 8. [Data Augmentation](#8)
#
# ## 9. [Extended Model](#9)
#
# ### *9.1. [Convolutional neural networks (CNN)](#9.1)*
#
# ### *9.2. [Hyperparameter tuning CNN](#9.2)*
#
# ### *9.3. [Simple MLP Deep Learning Model](#9.3)*
#
# ## 10. [Extra data](#10)
#
# ### *10.1. [Extra data Cell Type Predictions](#10.1)*
#
# ### *10.2. [Retrain model on the combined data](#10.2)*
#
# ## 11. [Best Model](#11)
#
# ## 12. [Independent Evaluation](#12)
#
# ## 13. [References](#13)
#
# <br>
#

# %% [markdown]
# <a id="1"></a>
# # 1. Import

# %%
import warnings
from kerastuner.tuners import RandomSearch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
from keras_tuner import RandomSearch
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint  # save best model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
import zipfile
!pip install keras-tuner

# read zip file

# data manipulation

# suppress informational messages from Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Visualization
%matplotlib inline
# override default style
sns.set_style('whitegrid')
# set default figure size
plt.rcParams['figure.figsize'] = (12, 8)
# set default font size
plt.rcParams['font.size'] = 14
# set default color palette
plt.rcParams['axes.prop_cycle'] = plt.cycler(
    color=['#ee4035', '#f37736', '#fdf498', '#7bc043', '#0392cf'])
# set default context
sns.set_context('notebook')
# set default color map
plt.rcParams['image.cmap'] = 'viridis'
# overwrite maplotlib to use seaborn defaults
sns.set()
# define color palette
palette = sns.color_palette()
# define color cycle
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# set default color
default_color = 'pink'

# ML


# Modelling

# Evaluation

# Define constants
BATCH_SIZE = 32
EPOCHS = 10
INPUT_DIM = (27, 27, 3)
HIDDEN_LAYER_DIM = 256
OUTPUT_CLASSES_ISCANCEROUS = 2
OUTPUT_CLASSES_CELLTYPE = 4

# Set the random seed for reproducibility
SEED = 2002
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# suppress warnings
warnings.filterwarnings('ignore')

# %%
AUTOTUNE = tf.data.experimental.AUTOTUNE

tf.__version__

# %% [markdown]
# <a id="2"></a>
# # 2. Data Overview
#
# The dataset consists of the images (`patch_images` folder) and a csv files as labels. The labels and the image names are in the CSV file. We will add the image path to the dataframe
#
# First, we extract the original zip file to the base directory

# %%
# extract the zip file
with zipfile.ZipFile('../Image_classification_data.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

# Define the base directory for the project
base_dir = 'Image_classification_data'

# Define the file paths for the CSV files
main_data_csv = os.path.join(base_dir, 'data_labels_mainData.csv')
extra_data_csv = os.path.join(base_dir, 'data_labels_extraData.csv')

# Read the main data CSV file
main_data = pd.read_csv(main_data_csv)
# Read the extra data CSV file
extra_data = pd.read_csv(extra_data_csv)

# number of rows and columns in the main data DataFrame and extra data DataFrame
print(
    f'\nThe main data DataFrame has {main_data.shape[0]} rows and {main_data.shape[1]} columns')
print(
    f'The extra data DataFrame has {extra_data.shape[0]} rows and {extra_data.shape[1]} columns')

# display the different columns in the main data DataFrame and extra data DataFrame
print(
    f'\nThe main data DataFrame has the following columns: {main_data.columns}')
print(
    f'The extra data DataFrame has the following columns: {extra_data.columns}')

# the first 3 rows of the main data DataFrame
main_data.head(3)

# %% [markdown]
# ### ---------> OBSERVATION:
#
# > The information provided suggests two datasets: main data and extra data. There are images (e.g., 22405.png) of cells taken from patients (e.g., patientID 1), with specific cell types (e.g., fibroblast) and an indication of whether the cell is cancerous or not (e.g., not cancerous in this case). Unlike the main data, the **extra data does not have** information on `cellTypeName` and `cellType`.
#
# > We have more than 10,000 rows for main and extra data despite having only 99 different patients because each patient likely has multiple images associated with them. Each row in the dataset represents a cell's individual image (27x27 RGB). Since histopathology imaging can capture many cells from a single tissue sample, having multiple images per patient is common. Each of these images might display different cells, cell types, or cell characteristics, contributing to the variability in the dataset. This variability helps train the machine learning model to recognize different features and patterns in the data, ultimately improving its classification performance.

# %% [markdown]
# #### Map image paths to the dataframes

# %%
# Read the main data CSV file
main_data = pd.read_csv(main_data_csv)

# Read the extra data CSV file
extra_data = pd.read_csv(extra_data_csv)

# Define the patch images folder
patch_images_folder = os.path.join(base_dir, 'patch_images')

# Map the ImageName column to the image file paths in the 'patch_images' folder
main_data['ImageFilePath'] = main_data['ImageName'].apply(
    lambda x: os.path.join(patch_images_folder, x))
extra_data['ImageFilePath'] = extra_data['ImageName'].apply(
    lambda x: os.path.join(patch_images_folder, x))

# # Display the main data DataFrame
# print(main_data.head())

# # Display the extra data DataFrame
# print(extra_data.head())

# the first 3 rows of the main data DataFrame
main_data.head(3)

# %% [markdown]
# ### --------------> OBSERVATION:
#
# > We add an additional column called `ImageFilePath` added to the main and extra data. The `ImageFilePath` column contains the file paths of the images in the `patch_images` folder. This lets us directly access and load the images using their file paths.
#
# > Now, each row has information about the cell and its cancer status and includes the file path to the corresponding image. This makes loading and processing the images easier for further analysis or input into a machine-learning model.

# %% [markdown]
# <a id="3"></a>
#
# # 3. Exploratory Data Analysis (EDA)
#
# + EDA for image processing involves visualizing and analyzing the images and their associated data to gain insights that can help improve the performance of machine learning models.
#
# + Once we have performed EDA on our image dataset, we can use the insights to preprocess the data, engineer new features, and design an appropriate machine-learning model for the given problem.

# %% [markdown]
# <a id="3.1"></a>
# ## 3.1 Correlation between variables
#
# Analyze the correlation between variables (e.g., `cellTypeName` and `cellType`) to identify potential patterns or relationships useful during feature engineering or model training.

# %%
# cellType and and CellTypeName are the same?
print(
    f'Are cellType and cellTypeName the same? {len(main_data.groupby(["cellType", "cellTypeName"]).size().reset_index(name="count"))}')

# cellType values for each cellTypeName
print(
    f'\nThe cellType values for each cellTypeName in the main data: \n{main_data.groupby(["cellTypeName"])["cellType"].unique().reset_index(name="cellType")}')

# correlation between 'cellTypeName' & 'cellType'
print("\nThe correlation between 'cellTypeName' and 'cellType' is:")
main_data[['cellTypeName', 'cellType']].corr()

# %% [markdown]
# ### -------------------> OBSERVATION:
#
# > In the case of image classification tasks using neural networks and deep learning models, having two columns with similar meanings ('cellTypeName' and 'cellType') is usually not a direct problem for the model. This is because the input to these models is the images themselves, not the tabular features like 'cellTypeName' or 'cellType'. The model learns to extract relevant features directly from the image data.
#
# However, these columns can still be relevant in other aspects of the project:
# + **Labels:** If one of these columns is used as the target label for the classification task, choosing the appropriate column that best represents the problem we are trying to solve is essential. Using the wrong or unclear label could lead to poor model performance or incorrect predictions.
# + **Data Splitting:** When splitting our data into training, validation, and testing sets, it is important to ensure that the distribution of labels in each set is representative of the overall dataset. Consider their relationship when splitting the data if the two columns are related, and we need to use both for different classification tasks.
# + **Model Evaluation:** When comparing different models or evaluating the performance of a single model, it is crucial to use the appropriate labels for each task. If the two columns have similar meanings but represent different classification tasks, we should evaluate our models accordingly.
#
# While these columns may not directly impact the performance of image-based neural networks and deep learning models, it is still essential to understand their relationship and relevance to the problem we are trying to solve. Please choose the appropriate labels and consider their relationship when working with data splitting and model evaluation.

# %% [markdown]
# <a id="3.2"></a>
# ## 3.2 Analyze and Confirm image dimensions
#
# Check the dimensions (height, width, and channels) of the images in the dataset. This can help identify any inconsistencies or outliers in the data that may need preprocessing or resizing.

# %%
# Analyze image dimensions
# Define a function to get the image dimensions


def get_image_dimensions(image_file_path):
    """
    Returns the dimensions of an image

    Parameters:
        image_file_path (str): The file path of the image

    Returns:
        tuple: A tuple containing the image dimensions
    """
    # Open the image using PIL
    image = Image.open(image_file_path)
    # Get the image dimensions
    image_width, image_height = image.size
    # Close the image
    image.close()
    # Return the image dimensions
    return image_width, image_height


# Get the image dimensions for the main data DataFrame
print(
    f'The unique image dimensions for the main data DataFrame are: {main_data["ImageFilePath"].apply(get_image_dimensions).unique()}')

# Get the image dimensions for the extra data DataFrame
print(
    f'The unique image dimensions for the extra data DataFrame are: {extra_data["ImageFilePath"].apply(get_image_dimensions).unique()}')


# %% [markdown]
# ### -----------> OBSERVATION:
#
# > This means that all the images in the main data DataFrame and the extra data DataFrame have consistent dimensions of 27x27 pixels. This is a good sign, as it indicates no inconsistencies or outliers in terms of image dimensions within the dataset. We can proceed with further data preprocessing and modeling without worrying about resizing or adjusting the images for consistency.

# %% [markdown]
# <a id="3.3"></a>
# ## 3.3 Analyze class distribution
#
# Check the distribution of classes/categories in the dataset to identify any class imbalance. This can help inform strategies for data augmentation, resampling, or using weighted loss functions during model training.
#

# %%

# cellTypeNames for patients negative for cancer
print(
    f'The cellTypeNames for patients negative for cancer are {main_data[main_data["isCancerous"] == 0]["cellTypeName"].unique()}\n')
# cellTypeNames for patients positive for cancer
print(
    f'The cellTypeNames for patients positive for cancer are {main_data[main_data["isCancerous"] == 1]["cellTypeName"].unique()}')

# Plot the distribution of isCancerous labels
plt.figure(figsize=(6, 4))
sns.countplot(x='isCancerous', data=main_data)
plt.title('Distribution of IsCancerous Labels')
plt.show()
print(
    f'The number of unique isCancerous labels is {len(main_data["isCancerous"].unique())}')

# Plot the distribution of cellType labels
plt.figure(figsize=(8, 4))
sns.countplot(x='cellTypeName', data=main_data)
plt.title('Distribution of Cell Type Labels')
plt.show()
# print cellType labels
print(
    f'The number of unique cellType labels is {len(main_data["cellTypeName"].unique())}')

# Plot the distribution of cellType labels for cancerous and non-cancerous cells
plt.figure(figsize=(8, 4))
sns.countplot(x='cellTypeName', hue='isCancerous', data=main_data)
plt.title('Distribution of Cell Type Labels by IsCancerous')
plt.show()

# print cellType labels for cancerous and non-cancerous cells
print(
    f'The number of unique cellType labels for cancerous cells is {len(main_data[main_data["isCancerous"] == 1]["cellTypeName"].unique())}\n')

# print the number of cancerous and non-cancerous cells for each cellType label
print("The table shows the distribution of cell types and their corresponding isCancerous labels:")
print(main_data.groupby(['cellTypeName', 'isCancerous']
                        ).size().reset_index(name='count'))

# %% [markdown]
# ### -------------------> OBSERVATIONS:
#
# > In the provided dataset, all cancerous cells are of the epithelial type, and there is only one unique cell type (epithelial) for cancerous cells.
#
# > **Interpretation of imbalanced data:**
# Our dataset is imbalanced, meaning the number of samples for each class is unequal. The epithelial class has more samples than the other classes. Moreover, the total number of non-cancerous cell is greater than cancerous cell. This imbalance can lead to biased models performing poorly on the under-represented classes, as the model will predict the majority class. Dealing with imbalanced data:
#
# >> + **Undersampling:** Remove some samples from the majority class (cancerous cells) to balance the class distribution. This method can lead to losing valuable information if not done carefully.
# >> + **Oversampling:** Duplicate samples from the minority classes (non-cancerous cells) or create synthetic samples using techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the class distribution. This can increase the risk of overfitting.
# Cost-sensitive learning: Assign different weights to different classes during model training, giving more importance to the minority classes.
# >> + **Ensemble methods:** Combine multiple models to improve classification performance. For example, use bagging or boosting with different sampling strategies.
# >> + **Evaluation metric selection:** Choose evaluation metrics that are more sensitive to imbalanced data, such as precision, recall, F1-score, or the area under the ROC curve (AUC-ROC).
#

# %%
# see if the same patient has the same cellType label in the main data DataFrame
print(
    f'The number of unique patients in the main data DataFrame is {len(main_data["patientID"].unique())}')

# Group the data by patientID and count the unique cellTypeName values for each patient
unique_cell_type_counts = main_data.groupby(
    'patientID')['cellTypeName'].nunique()
print(
    f'The number of unique patients with the same cellType label in the main data DataFrame is {len(unique_cell_type_counts[unique_cell_type_counts > 1])}')

# list of patient IDs with the different cellType label
patient_ids = unique_cell_type_counts[unique_cell_type_counts > 1].index.tolist(
)
print(f'The patient IDs with the different cellType label are: {patient_ids}')


# %% [markdown]
# ### ------------------> OBSERVATIONS:
#
# > The main dataset description contains 27x27 RGB images of cells from 60 different patients. Repeated data for the same patient means multiple images are taken from the same patient. This can introduce a bias in the dataset, as the samples from the same patient may be more similar to each other than samples from different patients. This can lead to overfitting and poor generalization to new patients.
# Dealing with repeated data for the same patient:
# >> + Patient-wise train-test split: Instead of randomly splitting the dataset into training and test sets, we can split it based on patients. This ensures that all samples from the same patient are either in the training set or the test set, but not in both. This reduces the risk of overfitting and provides a more reliable evaluation of the model's performance.
# >> + Cross-validation: Perform k-fold cross-validation using patient-wise splits. This can help us better to estimate the model's performance on new patients and reduce the risk of overfitting.
# >> + Data augmentation: Apply various data augmentation techniques, such as rotation, flipping, or scaling, to create new samples from the existing data. This can increase the diversity of the dataset and reduce the impact of repeated samples from the same patient.

# %% [markdown]
# <a id="3.4"></a>
# ## 3.4 Visualize sample images
#
# + Randomly select and display a few images from each class or category to understand the visual characteristics and variations in the data.
#
# + Take a look of cancerous cell and non-cancerous cell to get the better sense how they look like

# %% [markdown]
# ### Cancerous Cell Images

# %%

# Filter cancerous cells
cancerous_cells = main_data[main_data['isCancerous'] == 1]

# Display a sample of cancerous cell images
num_images_to_display = 9
random_samples = random.sample(
    list(cancerous_cells['ImageFilePath']), num_images_to_display)

plt.figure(figsize=(10, 10))

for index, image_path in enumerate(random_samples):
    img = Image.open(image_path)

    plt.subplot(3, 3, index + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image: {os.path.basename(image_path)}")
    # add the CellTypeName as the title
    plt.title(
        f"Image: {os.path.basename(image_path)}\nCell Type: {cancerous_cells[cancerous_cells['ImageFilePath'] == image_path]['cellTypeName'].values[0]}")

plt.suptitle('9 Sample Cancerous Cell Images', fontsize=16)
plt.show()


# %% [markdown]
# ### Non-Cancerous Cell Images

# %%
# Filter non-cancerous cells
non_cancerous_cells = main_data[main_data['isCancerous'] == 0]

# Display a sample of non-cancerous cell images
num_images_to_display = 9
random_samples = random.sample(
    list(non_cancerous_cells['ImageFilePath']), num_images_to_display)

plt.figure(figsize=(10, 10))

for index, image_path in enumerate(random_samples):
    img = Image.open(image_path)

    plt.subplot(3, 3, index + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image: {os.path.basename(image_path)}")
    # add the CellTypeName as the title
    plt.title(
        f"Image: {os.path.basename(image_path)}\nCell Type: {non_cancerous_cells[non_cancerous_cells['ImageFilePath'] == image_path]['cellTypeName'].values[0]}")

plt.suptitle('9 Sample Non-Cancerous Cell Images', fontsize=16)
plt.show()


# %% [markdown]
# <a id="3.5"></a>
# ## 3.5 Analyze color distributions
#
# Plot the color distribution (e.g., RGB histograms) for a sample of images to understand the range of colors and their intensities. This can help identify potential issues related to color balancing or the need for data augmentation.

# %%
# plot the color distribution (RGB histograms)
def plot_color_distribution(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)

    # plot the RGB histograms
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    ax[0].hist(img_array[:, :, 0].flatten(), bins=256,
               color='red', label='Red', alpha=0.5)
    ax[0].set_title('Red Histogram')
    ax[0].legend()
    ax[1].hist(img_array[:, :, 1].flatten(), bins=256,
               color='green', label='Green', alpha=0.5)
    ax[1].set_title('Green Histogram')
    ax[1].legend()
    ax[2].hist(img_array[:, :, 2].flatten(), bins=256,
               color='blue', label='Blue', alpha=0.5)
    ax[2].set_title('Blue Histogram')
    ax[2].legend()

    plt.suptitle(
        f'Color Distribution for Image: {os.path.basename(image_path)}', fontsize=16)
    plt.show()


# plot the color distribution for a cancerous cell
plot_color_distribution(cancerous_cells['ImageFilePath'].values[0])

# plot the color distribution for a non-cancerous cell
plot_color_distribution(non_cancerous_cells['ImageFilePath'].values[0])


# %% [markdown]
# <a id="3.6"></a>
# ## 3.6 Check for duplicated images
#
# Identify and remove any duplicate images in the dataset to ensure that the model does not overfit to specific instances.

# %%
# Check for duplicate images
print(
    f'The number of duplicate images is {len(main_data[main_data.duplicated(subset=["ImageFilePath"])])}')

# %% [markdown]
# <a id="4"></a>
# # 4. Patient-wise train-test-validation split
#
# + We use hold-out validation technique. In hold-out validation, we divide the dataset into separate sets: training, validation, and test. The model is trained on the training set, hyperparameters are tuned using the validation set, and the final model performance is evaluated on the test set.
#
# + Instead of randomly splitting the dataset into training and test sets, we can split it based on patients. This ensures that all samples from the same patient are either in the training or the test set, but not in both. This reduces the risk of overfitting and provides a more reliable evaluation of the model's performance.
#
# + We might introduce data leakage when we use a random train-test split **without** considering the patient information. The model can memorize specific features or patterns from a patient seen in the training set and perform better on the test set than on unseen patients. This can lead to an overly optimistic evaluation of the model's performance.
# So, a patient-wise train-test split is recommended for this medical data, as it helps to assess the model's ability to generalize to new, unseen patients, which is crucial for real-world deployment.
#
# **Steps:**
#
# 1. **Split into two sets:** train_data (80% of the data) and test_data (20% of the data). The train_test_split function from the scikit-learn library is used to perform this split. The stratify parameter is set to main_data[`patientID`], ensuring that patient IDs' distribution remains similar across both the training and test sets. This helps maintain the dataset's representativeness and reduces the chance of introducing sampling bias. The random_state parameter is set to a fixed value (SEED) to ensure that the split is reproducible.
#
# 2. **Further split the train_data** into train_data (75% of the remaining data) and val_data (25% of the remaining data) using the same train_test_split function. Again, the stratify parameter is set to maintain a similar distribution of patient IDs across the two sets, and the random_state is set to ensure reproducibility.

# %%
# Split the main dataset into training (60% of the original data), validation (20% of the original data), and test (20% of the original data).
train_data, test_data = train_test_split(
    main_data, test_size=0.2, stratify=main_data['patientID'], random_state=SEED)
train_data, val_data = train_test_split(
    train_data, test_size=0.25, stratify=train_data['patientID'], random_state=SEED)

# print the shapes of the resulting DataFrames
print(f'The shape of the train_data DataFrame is {train_data.shape}')
print(f'The shape of the val_data DataFrame is {val_data.shape}')
print(f'The shape of the test_data DataFrame is {test_data.shape}\n')

# print the number of unique patients in each of the three DataFrames
print(
    f'The number of unique patients in the train_data DataFrame is {len(train_data["patientID"].unique())}')
print(
    f'The number of unique patients in the val_data DataFrame is {len(val_data["patientID"].unique())}')
print(
    f'The number of unique patients in the test_data DataFrame is {len(test_data["patientID"].unique())}')

# %% [markdown]
# <a id="5"></a>
# # 5. Data Preprocessing
#
# Preprocessing is done after splitting the data to prevent data leakage. Data leakage occurs when information from the validation or test sets is inadvertently incorporated into the training set during preprocessing. This can lead to overly optimistic model performance estimates and poorer generalization to new, unseen data. Additionally, certain preprocessing steps, such as data normalization or feature scaling, are calculated based on the training data's statistics (e.g., mean and standard deviation). Applying these calculations to the validation and test sets helps simulate the scenario where the model encounters unseen data. This ensures the model's performance metrics reflect its ability to generalize to new data.
#
# **Steps:**
#
# 1. **Loading and preprocessing images:** The `load_and_preprocess_images` function reads the image files, resizes them to 27x27, converts them to arrays, and normalizes the pixel values to be between 0 and 1. This function is applied to the training, validation, and test data.
#
# 2. **One-hot encoding of `cellType` labels:** The cellType labels are one-hot encoded using the OneHotEncoder. This is done for training, validation, and test datasets, resulting in train_cell_type_labels, val_cell_type_labels, and test_cell_type_labels.
#
# 3. **Handling `isCancerous` labels:** The code block includes commented-out lines for one-hot encoding the isCancerous labels, but it is not necessary for binary classification. Instead, the labels are reshaped into a 2D array, resulting in train_is_cancerous_labels, val_is_cancerous_labels, and test_is_cancerous_labels.

# %%
# Load images and normalize pixel values


def load_and_preprocess_images(df):
    images = []
    for image_path in df['ImageFilePath']:
        image = load_img(image_path, target_size=(27, 27))
        image = img_to_array(image)
        image = image / 255.0
        images.append(image)
    return np.array(images)


# load and preprocess the images in the main dataset
train_images = load_and_preprocess_images(train_data)
val_images = load_and_preprocess_images(val_data)
test_images = load_and_preprocess_images(test_data)
# One-hot encode the cellType
encoder = OneHotEncoder(sparse=False)
train_cell_type_labels = encoder.fit_transform(
    train_data['cellType'].values.reshape(-1, 1))
val_cell_type_labels = encoder.transform(
    val_data['cellType'].values.reshape(-1, 1))
test_cell_type_labels = encoder.transform(
    test_data['cellType'].values.reshape(-1, 1))
# not one-hot encoded for binary classification
train_is_cancerous_labels = train_data['isCancerous'].values.reshape(-1, 1)
val_is_cancerous_labels = val_data['isCancerous'].values.reshape(-1, 1)
test_is_cancerous_labels = test_data['isCancerous'].values.reshape(-1, 1)

# load and preprocess the images in the extra dataset
extra_images = load_and_preprocess_images(extra_data)
# not one-hot encoded for binary classification
extra_is_cancerous_labels = extra_data['isCancerous'].values.reshape(-1, 1)

# print the sample encoded labels in the main dataset
print(
    f'The example of one-hot encoded labels for the first image is {train_cell_type_labels[0]}')

# %% [markdown]
# <a id="6"></a>
# # 6. Evaluation Metrics
#
# + **Classification Accuracy:** This is the simplest form of evaluation metric for multiclass classification problems. It's calculated as the number of correct predictions made by the model divided by the total number of predictions.
#
# ```python
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(y_true, y_pred)
# ```
#
# + **[Accuracy by Class (Precision, Recall, F1-score for each class)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html):** In multiclass problems, it is often more insightful to measure the accuracy for each class separately, using metrics such as precision, recall, and [F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html). Precision is the proportion of true positives (i.e., the number of items correctly labeled as belonging to the positive class) among all positives, both true and false. Recall is the proportion of true positives among all actual positives. The F1 score is the harmonic mean of precision and recall. These metrics can be computed for each class, treating it as the positive class and all other classes as negatives.
#
# ```python
# from sklearn.metrics import classification_report
# print(classification_report(y_true, y_pred))
# ```
#
# + **[Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html):** A confusion matrix is a table that is used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known. Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa). The name stems from the fact that it makes it easy to see if the system is confusing two classes (i.e., commonly mislabeling one as another).
#
# ```python
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_true, y_pred)
# ```
#
# #### Imbalanced classes?
#
# When dealing with imbalanced datasets, accuracy can be misleading as a metric. The model might predict the majority class most of the time and still achieve a high accuracy score. Evaluating Accuracy by Class and the Confusion Matrix is more informative in these cases:
#
# + **Accuracy by Class:** This includes precision, recall, and F1-score for each class. These metrics consider both false positives and false negatives. F1-score, in particular, is a good choice when the data is imbalanced, as it seeks a balance between precision and recall. It gives us a better measure of the incorrectly classified cases than the Accuracy Metric.
#
# + **Confusion Matrix:** The confusion matrix is a handy presentation of the accuracy of a model with two or more classes. For each class, it shows the distribution of its instances classified into all classes. This allows us to see not just where the model is getting it right but, more importantly, where it is getting it wrong - which classes are being confused with each other and where the model's weaknesses lie.
# Therefore, both of these evaluation methods will give us a better insight into how well our model performs in each class, which is especially important when dealing with imbalanced datasets.
#
# While we can **use accuracy to get a sense of overfitting or underfitting**, we use better metrics for test set model performance in the case of imbalanced datasets. Precision, recall, F1-score, and AUC-ROC are better alternatives. For a cancer cell type classification problem, the criticality of each metric often depends on the specific context and use case. However, in general, due to the severe consequences of misclassification, particularly false negatives, the following metrics are often considered important:
#
# + **Precision (Positive Predictive Value):** This is the proportion of true positives out of all positive results (both true positives and false positives). High precision means that it is likely correct when the model predicts a cell as cancerous. In a cancer detection context, precision is crucial to avoid overdiagnosis and unnecessary treatments that can have harmful side effects and increase healthcare costs.
#
# + **Recall (Sensitivity or True Positive Rate):** This is the proportion of true positives out of all actual positive cases (true positives and false negatives). High recall means the model correctly identifies a high proportion of all cancerous cells. In a cancer detection context, recall is particularly critical because failing to identify a cancerous cell (false negative) can lead to delayed or missed treatment, potentially leading to severe health consequences.
#
# + **F1-Score:** This is the harmonic mean of precision and recall, and it tries to balance these two metrics. It is particularly useful if there is an uneven class distribution, as precision and recall might give misleading results in such a case.
#
# + **AUC-ROC (Area Under the Receiver Operating Characteristics Curve):** This metric considers the trade-off between sensitivity (true positive rate) and specificity (true negative rate). An AUC-ROC of 1.0 indicates perfect classification, while an AUC-ROC of 0.5 indicates a model that is no better than random guessing. This metric is widely used in binary classification problems and can indicate the model's performance across different thresholds.
#

# %%
# evaluate using precision, recall, and f1-score, auc, and confusion matrix


def evaluate_and_report(test_is_cancerous_labels, is_cancerous_predictions, test_cell_type_labels, cell_type_predictions, is_cancerous_nn_model, cell_type_nn_model, is_cancerous_model_name, cell_type_model_name):
    is_cancerous_precision, is_cancerous_recall, is_cancerous_f1, _ = precision_recall_fscore_support(
        test_is_cancerous_labels, is_cancerous_predictions, average='binary')
    cell_type_precision, cell_type_recall, cell_type_f1, _ = precision_recall_fscore_support(
        test_cell_type_labels.argmax(axis=1), cell_type_predictions, average='macro')

    print(
        f'\nIs Cancerous Precision: {round(is_cancerous_precision * 100, 2)}%, Recall: {round(is_cancerous_recall * 100, 2)}%, F1-score: {round(is_cancerous_f1 * 100, 2)}%')
    print(
        f'Cell Type Precision: {round(cell_type_precision * 100, 2)}%, Recall: {round(cell_type_recall * 100, 2)}%, F1-score: {round(cell_type_f1 * 100, 2)}%')

    # Compute ROC AUC score
    is_cancerous_auc_score = roc_auc_score(
        test_is_cancerous_labels, is_cancerous_predictions, average='macro')
    lb = LabelBinarizer()
    test_cell_type_labels_2d = lb.fit_transform(test_cell_type_labels)
    cell_type_predictions_2d = lb.transform(cell_type_predictions)
    cell_type_auc_score = roc_auc_score(
        test_cell_type_labels_2d, cell_type_predictions_2d, average='macro')

    print(f"\nIs Cancerous ROC AUC Score: {is_cancerous_auc_score}")
    print(f"Cell Type ROC AUC Score: {cell_type_auc_score}\n")

    # Create a DataFrame from the classification report
    report_df = pd.DataFrame({
        'precision': [is_cancerous_precision, cell_type_precision],
        'recall': [is_cancerous_recall, cell_type_recall],
        'f1-score': [is_cancerous_f1, cell_type_f1],
        'auc': [is_cancerous_auc_score, cell_type_auc_score],
        'model_name': [is_cancerous_model_name, cell_type_model_name],
        'model_params': [str(is_cancerous_nn_model.get_config()), str(cell_type_nn_model.get_config())]
    })

    return report_df

# plot confusion matrix


def plot_confusion_matrix(labels, predictions, title):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d",  cmap="pink", cbar=False)
    plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

# plot learning curve with loss and accuracy side by side


def plot_learning_curve(history, title):
    plt.figure(figsize=(15, 5))

    # plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.suptitle(title)
    plt.show()

# %% [markdown]
# <a id="7"></a>
# # 7. Baseline Model - Traditional Neural Network (not CNN)
#
# A baseline model is a simple, often traditional, machine learning model that is easy to understand and implement. Despite its simplicity, it is expected to perform reasonably well on the problem. The performance of this baseline model then serves as a benchmark for other models. Any advanced model we develop must outperform this baseline to be considered valuable. In the case of our study, we chose a traditional neural network as our baseline model. Several reasons drove this decision.
#
# + Firstly, neural networks have been successfully applied in various classification tasks, demonstrating their robustness and versatility. Despite being relatively simple compared to more recent variations and architectures, traditional neural networks can capture complex patterns in data, making them a reasonable first choice for our task.
#
# + Secondly, the structure of our chosen neural network is straightforward, comprising a couple of dense layers interspersed with dropout layers for regularization. This simplicity aids in the interpretability of the model, allowing us to understand how the model works and what aspects might need improving.
#
# + Lastly, this traditional neural network is a stepping stone for more complex models. By understanding its performance and limitations, we gain insights to guide the development and fine-tuning of more advanced models, such as convolutional neural networks, for our cancer cell-type classification task. Therefore, the traditional neural network model serves as our baseline, helping us to establish a performance benchmark and gain critical insights that inform the development of more sophisticated models.
#
# The chosen architecture for the baseline model is a simple, traditional Neural Network (also known as a Feedforward Neural Network). There are several reasons for choosing this architecture:
#
# + **Simplicity:** The architecture is straightforward. It uses a series of dense (fully connected) layers with a non-linear activation function (ReLU - Rectified Linear Unit), which allows the model to learn complex patterns in the data. The simplicity of the architecture makes it easy to implement and debug, providing a good starting point for our experiments.
# + **Versatility:** Despite its simplicity, this architecture can model various problems, including the binary and multi-class classification problems we are dealing with in this project. It is a versatile choice that can give us a good performance on our task.
# + **Regularization:** The model includes dropout layers, a simple and effective regularization method to prevent overfitting. During training, dropout layers randomly set a fraction of input units to 0 at each update, which helps to prevent overfitting.
# + **Scalability:** The model is easily scalable. If necessary, we can easily make the model more complex by adding more layers or units in each layer without changing the overall architecture too much.
# + **Interpretability:** Despite being a neural network, this architecture is still relatively interpretable. We can examine the model's weights to understand what the model is learning.
# Therefore, this architecture balances complexity and performance, making it a suitable choice for a baseline model.


# %%
% % time
# Define the architecture for the traditional neural network for binary classification (is_cancerous)


def create_is_cancerous_nn_model(input_shape):
    model = Sequential(name='is_cancerous_nn_model')
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


is_cancerous_nn_model = create_is_cancerous_nn_model(train_images.shape[1:])
is_cancerous_nn_model.compile(
    optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Define the architecture for the traditional neural network for multi-class classification (cell type)


def create_cell_type_nn_model(input_shape, num_classes):
    model = Sequential(name='cell_type_nn_model')
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


cell_type_nn_model = create_cell_type_nn_model(
    train_images.shape[1:], train_cell_type_labels.shape[1])
cell_type_nn_model.compile(
    optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the traditional neural network models
is_cancerous_nn_history = is_cancerous_nn_model.fit(train_images, train_is_cancerous_labels, validation_data=(
    val_images, val_is_cancerous_labels), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
cell_type_nn_history = cell_type_nn_model.fit(train_images, train_cell_type_labels, validation_data=(
    val_images, val_cell_type_labels), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

# Save the traditional neural network models
is_cancerous_nn_model.save('models/is_cancerous_nn_model.h5')
cell_type_nn_model.save('models/cell_type_nn_model.h5')
print("Traditional Neural Network models saved successfully!\n")

# Evaluate the models using precision, recall, F1-score, and support
is_cancerous_predictions = is_cancerous_nn_model.predict(test_images)
is_cancerous_predictions = np.where(is_cancerous_predictions > 0.5, 1, 0)
cell_type_predictions = cell_type_nn_model.predict(test_images)
cell_type_predictions = np.argmax(cell_type_predictions, axis=1)

# plot learning curves
plot_learning_curve(is_cancerous_nn_history,
                    'Traditional Neural Network (Is Cancerous) Learning Curve')
plot_learning_curve(cell_type_nn_history,
                    'Traditional Neural Network (Cell Type) Learning Curve')

# plot confusion matrices
plot_confusion_matrix(test_is_cancerous_labels, is_cancerous_predictions,
                      'Traditional Neural Network (Is Cancerous) Confusion Matrix')
plot_confusion_matrix(test_cell_type_labels.argmax(
    axis=1), cell_type_predictions, 'Traditional Neural Network (Cell Type) Confusion Matrix')

# store the traditional neural network report in a DataFrame
traditional_nn_report_df = evaluate_and_report(test_is_cancerous_labels, is_cancerous_predictions, test_cell_type_labels, cell_type_predictions,
                                               is_cancerous_nn_model, cell_type_nn_model, 'Traditional Neural Network (Is Cancerous)', 'Traditional Neural Network (Cell Type)')
traditional_nn_report_df

# %% [markdown]
# ### ---------> OBSERVATION:
#
# Our analysis of the performance of two traditional neural network models, each designed for different tasks, revealed a stark contrast in their behaviors.
#
# #### Is Cancerous Model:
# >> The first model, "Traditional Neural Network (Is Cancerous)," demonstrated a significant discrepancy between its training and validation performance. While it achieved high accuracy during training, the corresponding validation accuracy was noticeably lower. The loss function also showed high values, indicating that the model might struggle to fit the training data. Interestingly, the validation performance was better than the training performance. The model showed promising results for the test set, with a precision of 0.840686, recall of 0.84901, F1-score of 0.844828, and an AUC of 0.869044. This disparity between training and validation performance suggests that the model might be underfitting the training data, implying that it could benefit from a more complex architecture or additional training.
#
# >> The confusion matrix shows that the model predicts all instances as 'not cancerous', which indicates that it is highly biased towards the negative class. This is also reflected in the precision, recall, and F1-score, which are all 0%. The ROC AUC score of 0.5 indicates that the model has no discriminative power between the classes.
# The confusion matrix for  `is_cancerous` classifier:
# ```
# [[1172    0]
#  [ 808    0]]
# ```
# >> This matrix tells us that our model predicted 1172 instances correctly as non-cancerous (True Negatives). However, it failed to correctly predict any cancerous instances, as indicated by the 808 instances incorrectly predicted as non-cancerous (False Negatives). There were no False Positives or True Positives. This indicates that the model needs to predict cancerous cells better, as it failed to predict any correctly.
#
# #### Cell Type Model
#
# >> On the other hand, the "Traditional Neural Network (Cell Type)" model showed a different pattern. The validation accuracy was lower than the training accuracy, and the model suffered from high loss values worse than those seen during training. This indicates that the model might need to be more balanced with the training data, which results in poor generalization of unseen data. The performance on the test set for this model was less satisfactory, with precision, recall, F1-score, and AUC values of 0.102020, 0.25000, 0.144907, and 0.500000, respectively. This performance dip on the test set is a clear sign of overfitting, and solutions might involve simplifying the model, introducing regularization, or augmenting the training data.
#
# >> This model is also highly biased, predicting all instances as belonging to one class. This is reflected in the low precision, recall, and F1-score values. The ROC AUC score of 0.5 also suggests that the model has no discriminative power between the classes.
# The confusion matrix for the  `cell_type` classifier looks like this:
# ```
# [[  0   0 366   0]
#  [  0   0 525   0]
#  [  0   0 808   0]
#  [  0   0 281   0]]
# ```
# >> This matrix tells us that our model consistently predicts one class (the third class) for all instances, regardless of their true class. For every row, the third column has all the predictions (366, 525, 808, and 281). This indicates a severe problem as the model must be more effective in distinguishing between classes. It is effectively predicting the same class for all instances.
#
#
# #### Recommendations to Improve Performance
# + **Modify Model Architecture:** We could make the model more complex by adding more layers or neurons in each layer. We could also try a different model type, such as a Convolutional Neural Network (CNN), which is typically good for image data.
# + **Optimization Algorithm:** Experiment with different optimization algorithms and their parameters. For example, we could try SGD, RMSprop, or Adam and adjust their learning rates.
# + **Early Stopping and Checkpoints:** Implement early stopping to save the model weights when the validation performance is best. This can prevent the model from overfitting to the training data.
# + **Data Augmentation:** We can artificially increase its size by creating modified versions of the images (e.g., rotated, zoomed, shifted). This can help improve the model's ability to generalize.
# + **Regularization:** If the model is overfitting, consider adding regularization techniques such as L1, L2, or dropout.
# + **Hyperparameter Tuning:** Perform systematic hyperparameter tuning using techniques like grid search or random search to find the optimal configuration for the models.
#

# %% [markdown]
# <a id="8"></a>
# # 8. Data Augmentation
#
# + We apply this technique on train data only (not validation or test set)
#
# + An ImageDataGenerator is created to apply data augmentation techniques such as rotation, shifting, shearing, zooming, and flipping. This helps increase the dataset's size and diversity, enabling the model to generalize better.

# %%
# Perform data augmentation
data_gen = ImageDataGenerator(rotation_range=20,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              vertical_flip=True,
                              fill_mode='nearest')

# %% [markdown]
# <a id="9"></a>
# # 9. Extended Model

# %% [markdown]
# <a id="9.1"></a>
# ## 9.1 Convolutional neural networks (CNN)
#
# The motivation to improve from the traditional Neural Network (NN) baseline model to a Convolutional Neural Network (CNN) model for this task comes from several factors:
#
# + **Ability to capture spatial features:** Images have a spatial structure, meaning the arrangement and ordering of pixels carry valuable information about the image's content. While traditional NNs treat inputs as a flat vector, ignoring this spatial structure, CNNs are designed to automatically and adaptively learn spatial hierarchies of features, which is advantageous for image data.
# + **Efficiency:** CNNs are more efficient for image data. In a fully connected layer, each neuron is connected to every neuron in the previous layer, and each connection has its own weight. This full connectedness leads to an explosion of connections and, thus, parameters to learn as the size of the image grows. In contrast, CNNs share weights across space, leading to fewer parameters, enabling the network to scale better to larger images.
# + **Feature extraction:** CNNs have a built-in capability for feature extraction, which means they can automatically learn and abstract features from raw data. This eliminates the need for manual feature extraction, which can be complex and time-consuming.
# + **Performance:** CNNs generally outperform traditional NNs on image tasks. They have been the go-to model for image classification tasks and have achieved state-of-the-art results on many image datasets.
#
# We define two baseline models using the [2D convolution layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D). Both models are convolutional neural networks (CNNs) for image classification tasks. They share a similar architecture, but the output layer and loss functions differ to accommodate their respective classification tasks.
#
# + **`create_is_cancerous_model`:** This model is designed for **binary classification** to determine whether an image is cancerous (1) or not (0). The architecture consists of three sets of convolutional layers, each followed by max-pooling layers. After these, a Flatten layer converts the 2D arrays into a 1D array for the Dense layers. The model has a fully connected hidden layer with 64 neurons and a ReLU activation function. A dropout layer with a rate of 0.5 is added to prevent overfitting. The final output layer has a single neuron with a sigmoid activation function, which is suitable for binary classification tasks. The model is compiled using the Adam optimizer, binary_crossentropy loss, and accuracy as the evaluation metric.
#
# + **`create_cell_type_model`:** This model is designed for **multi-class classification** to determine the cell type in the image. The architecture is similar to the is_cancerous_model with three sets of convolutional layers followed by max-pooling layers, a Flatten layer, and a fully connected hidden layer with 64 neurons and a ReLU activation function. It also includes a dropout layer with a rate of 0.5 to prevent overfitting. However, the output layer has 4 neurons with a softmax activation function, which is suitable for multi-class classification tasks (in this case, 4 different cell types). The model is compiled using the Adam optimizer, categorical_crossentropy loss, and accuracy as the evaluation metric.
#
#
# #### The choice of parameters for the CNN:
#
# > **Number of layers and filters:** The model starts with a smaller number of filters (32) and progressively increases this number in the deeper layers (64 and then 128). This is a common strategy in CNN design, based on the concept that the network should extract a hierarchical representation of the data. The initial layers capture simple and generic features such as edges and textures, while the deeper layers capture more complex and abstract features that are more specific to the task of cancer image classification. The increase in the number of filters allows the model to capture a larger number of these complex features.
#
# > **Filter size (3x3):** The choice of 3x3 filters is quite common in successful CNN architectures, such as VGGNet, due to their efficiency in capturing local and spatial features in an image. In the context of cancer image classification, these small filters are able to capture fine-grained details, which can be crucial in differentiating between cancerous and non-cancerous cells, as well as different types of cells.
#
# > **Input channels:** The number of input channels for the first layer is set to 3, corresponding to the RGB color channels of the input images. This allows the model to capture color information, which can be significant in cancer image classification - for example, cancerous cells may exhibit different color characteristics compared to healthy cells. For the subsequent layers, the number of input channels equals the number of filters in the previous layer, as the output of each filter serves as a separate input channel for the next layer.
#
# > **[Max Pooling layers (2x2)](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D):** Max Pooling layers are utilized to progressively reduce the spatial dimensions of the input, thereby controlling the computational complexity of the model. This also provides a form of translation invariance, meaning the model can recognize cancerous patterns regardless of their location in the image.
#
# > **Activation function 'ReLU':** The ReLU activation function introduces non-linearity into the model without affecting the receptive fields of the convolution layers. This is particularly important for a task like cancer image classification, where the distinguishing features may not be linearly separable.
#
# > **Flattening layer:** The Flattening layer is used to convert the 2D feature maps into a 1D vector, enabling the output of the convolutional layers to be used as input to the Dense layers, which expect input in this format.
#
# > **Dense layer and Dropout:** The Dense layer with 64 neurons allows the model to learn non-linear combinations of the features obtained from the convolutional layers. The Dropout layer, which randomly sets a proportion of input units to 0 at each update during training time, is used to prevent overfitting. Overfitting is a common issue in machine learning tasks, and particularly in medical imaging tasks where the available data is often limited.
#
#
# #### The choice of optimizer
#
# [Adam (Adaptive Moment Estimation) optimizer](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) is a popular choice for training deep learning models because of its efficiency and low memory requirement. Adam has been extensively used in various deep learning tasks and has proven to be effective. Hence, it is a good starting point for most deep learning-based tasks, including the cancer image classification. Though Adam is a good default choice, depending on the specific task and dataset, other optimizers may sometimes perform better. Therefore, we will experiment with several optimizers when tuning a deep learning model later on.
#

# %% [markdown]
# #### Estimate Model Complexity - Number of parameters per Conv2D layer:
#
# + Formula: `number_parameters = (filter_height * filter_width * input_channels + 1 (for the bias)) * number_of_filters`
#
# **Note:** The "+1" term in the formula accounts for the bias term associated with each filter.

# %%
# First Conv2D layer
filter_height = 3
filter_width = 3
input_channels = 3  # RGB images
number_of_filters = 32

# Calculate number of parameters for the first Conv2D layer
number_parameters_first = (
    filter_height * filter_width * input_channels + 1) * number_of_filters
print(
    f'Number of parameters for the first Conv2D layer: {number_parameters_first}')

# Second Conv2D layer
input_channels = 32  # from the previous Conv2D layer
number_of_filters = 64

# Calculate number of parameters for the second Conv2D layer
number_parameters_second = (
    filter_height * filter_width * input_channels + 1) * number_of_filters
print(
    f'Number of parameters for the second Conv2D layer: {number_parameters_second}')

# Third Conv2D layer
input_channels = 64  # from the previous Conv2D layer
number_of_filters = 128

# Calculate number of parameters for the third Conv2D layer
number_parameters_third = (
    filter_height * filter_width * input_channels + 1) * number_of_filters
print(
    f'Number of parameters for the third Conv2D layer: {number_parameters_third}')


# %% [markdown]
# ### ---------------> OBSERVATIONS:
#
# > So, the total number of parameters in the Conv2D layers estimated to be **93248**
#
# > We assume that there is no padding ('valid' padding) and a stride of 1 in each dimension. If padding ('same' padding) or a different stride is used, the number of input channels to the next layer and hence the number of parameters could change.

# %% [markdown]
# #### Model Training
#
# Using preprocessed data, we train the two models, is_cancerous_model and cell_type_model.
#
# + **`is_cancerous_history`:** The is_cancerous_model is trained using the augmented training images (train_images) and corresponding binary labels (train_is_cancerous_labels). The data_gen.flow() function generates batches of augmented images and their labels on the fly during training. The batch_size is set using the BATCH_SIZE variable. The model is validated on the validation images (val_images) and their corresponding binary labels (val_is_cancerous_labels). The number of training iterations or epochs is set using the EPOCHS variable. The verbose parameter is set to 2, which means that only the most important information (such as the epoch number and validation loss) will be displayed during training. The training history is stored in the is_cancerous_history variable.
#
# + **`cell_type_history`:** The cell_type_model is trained using the augmented training images (train_images) and corresponding multi-class labels (train_cell_type_labels). Similar to the is_cancerous_model, the data_gen.flow() function generates batches of augmented images and their labels on-the-fly during training. The batch_size is set using the BATCH_SIZE variable. The model is validated on the validation images (val_images) and their corresponding multi-class labels (val_cell_type_labels). The number of training iterations or epochs is set using the EPOCHS variable. The verbose parameter is set to 2, displaying only the most important information during training. The training history is stored in the cell_type_history variable.

# %%
% % time
# Create the CNN model


def create_is_cancerous_model():
    model = Sequential(name='is_cancerous_model')
    # The first convolution
    # The input shape is the desired size of the image 27x27 with 3 bytes color (RGB)
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_DIM))
    # downsampling the output of the first convolutional layer by 2
    model.add(MaxPooling2D((2, 2)))
    # The second convolution
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    # The third convolution
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    # Flatten the 2D arrays for fully connected layers (Dense layers) after the convolutional layers and pooling layers
    model.add(Flatten())
    # 64 neuron hidden layer
    model.add(Dense(64, activation='relu'))
    # 50% dropout to prevent overfitting
    model.add(Dropout(0.5))
    # Sigmoid activation function is used for binary classification. 1 Output neuron ranges from 0 to 1 (0 = not cancerous, 1 = cancerous)
    model.add(Dense(1, activation='sigmoid'))
    return model


is_cancerous_model = create_is_cancerous_model()
is_cancerous_model.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model for cell-type classification CNN model


def create_cell_type_model():
    model = Sequential(name='cell_type_model')
    # The first convolution
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_DIM))
    # downsampling the output of the first convolutional layer by 2
    model.add(MaxPooling2D((2, 2)))
    # The second convolution
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    # The third convolution
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    # Flatten the 2D arrays for fully connected layers (Dense layers) after the convolutional layers
    model.add(Flatten())
    # 64 neuron hidden layer
    model.add(Dense(64, activation='relu'))
    # 50% dropout to prevent overfitting
    model.add(Dropout(0.5))
    # Softmax activation function is used for multi-class classification. 4 Output neurons
    model.add(Dense(OUTPUT_CLASSES_CELLTYPE, activation='softmax'))
    return model


cell_type_model = create_cell_type_model()
cell_type_model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Fit the model
# Train the models using the preprocessed data
is_cancerous_history = is_cancerous_model.fit(data_gen.flow(train_images, train_is_cancerous_labels, batch_size=BATCH_SIZE),
                                              validation_data=(
                                                  val_images, val_is_cancerous_labels),
                                              epochs=EPOCHS, verbose=2)

cell_type_history = cell_type_model.fit(data_gen.flow(train_images, train_cell_type_labels, batch_size=BATCH_SIZE),
                                        validation_data=(
                                            val_images, val_cell_type_labels),
                                        epochs=EPOCHS, verbose=2)

# save the CNN models
is_cancerous_model.save('models/is_cancerous_cnn_model.h5')
cell_type_model.save('models/cell_type_cnn_model.h5')
print('Convolutional Neural Network models saved successfully!')

# Evaluate the model using the test data
is_cancerous_loss, is_cancerous_accuracy = is_cancerous_model.evaluate(
    test_images, test_is_cancerous_labels, verbose=2)
cell_type_loss, cell_type_accuracy = cell_type_model.evaluate(
    test_images, test_cell_type_labels, verbose=2)

print(f'is_cancerous_loss: {is_cancerous_loss}')
print(f'is_cancerous_accuracy: {is_cancerous_accuracy}')
print(f'cell_type_loss: {cell_type_loss}')
print(f'cell_type_accuracy: {cell_type_accuracy}')

# plot learning curves
plot_learning_curve(is_cancerous_history, 'is_cancerous_CNN_model')
plot_learning_curve(cell_type_history, 'cell_type_CNN_model')

# get predictions
is_cancerous_predictions = is_cancerous_model.predict(test_images)
# convert predictions to binary
is_cancerous_predictions = np.where(is_cancerous_predictions > 0.5, 1, 0)
cell_type_predictions = cell_type_model.predict(test_images)
# convert predictions to one-hot encoded vectors
cell_type_predictions = np.argmax(cell_type_predictions, axis=1)

# plot confusion matrices
plot_confusion_matrix(test_is_cancerous_labels,
                      is_cancerous_predictions, 'is_cancerous_CNN_model')
plot_confusion_matrix(test_cell_type_labels.argmax(
    axis=1), cell_type_predictions, 'cell_type_CNN_model')


cnn_report_df = evaluate_and_report(test_is_cancerous_labels, is_cancerous_predictions, test_cell_type_labels,
                                    cell_type_predictions, is_cancerous_model, cell_type_model, 'CNN (Is Cancerous)', 'CNN (Cell Type)')
# concat the models to the report dataframe traditional_nn_report_df
evaluate_and_report_df = pd.concat(
    [traditional_nn_report_df, cnn_report_df], ignore_index=True)
evaluate_and_report_df

# %% [markdown]
# ### -----------> OBSERVATION:
#
# > Comparing our CNN models with the baseline traditional neural network model, we see a substantial performance improvement. For the `Is Cancerous` model, our Precision, Recall, F1-score, and ROC AUC score outperforms the corresponding metrics of the baseline model. Similarly, despite the inherent complexity of multi-class classification, our `Cell Type` model also shows considerable improvements over the baseline.
#
# > Our CNN models exhibit a good balance between bias and variance, demonstrating their ability to generalize well to unseen data. The dropout layers included in our models have effectively regularized the network, preventing overfitting by randomly setting a fraction of input units to 0 at each update during training time. This has resulted in a more robust model that performs well not just on our training data but also on our validation and test data.
#
#
# #### Is Cancerous Model
# >> The confusion matrix shows that the model is quite good at predicting cancerous and non-cancerous cases. With True Positives at 723 and True Negatives at 1056, the model correctly identifies many instances. However, there are still some mistakes, with 85 False Negatives and 116 False Positives.
# >> The evaluation metrics further confirm this: with a Precision of 86.17%, the model is correct 86.17% of the time when it predicts a case is cancerous. The Recall of 89.48% shows that it correctly identifies 89.48% of all actual cancerous cases. The F1-score, a combination of Precision and Recall, is 87.8%, indicating a good balance between Precision and Recall. The ROC AUC score is 0.8979, which is quite good, as it is close to 1.
#
# #### Cell Type Model
# >> The confusion matrix for this model shows a mixed picture. The model does well for some classes (like class 3, with 755 correct predictions) but not so well for others (like class 4, with only 53 correct predictions).
# >> The evaluation metrics reflect this mixed performance. The Precision is 66.18%, meaning the model is correct 66.18% of the time when it predicts a particular cell type. The Recall is 60.59%, indicating it correctly identifies 60.59% of all actual instances for each cell type. The F1-score is 60.58%, which is considerably lower than the Is Cancerous model, suggesting this model struggles more with balancing Precision and Recall. The ROC AUC score is 0.7506, which is lower than the Is Cancerous model, suggesting this model's accuracy in distinguishing between the different cell types could be better.
#
# > The Is Cancerous model performs better than the Cell Type model regarding the confusion matrix and the evaluation metrics. However, it is important to note that these are different tasks with different challenges: predicting whether a case is cancerous is binary, whereas predicting the cell type is a multi-class task, which is often more difficult.
#
# #### Recommendations to Improve Performance
#
# We would explore more options to optimize this CNN model by using hyperparameter tuning
#
#

# %%
# model architecture network summary for is_cancerous_model
is_cancerous_model.summary()

# %% [markdown]
# ### -------------> OBSERVATION:
#
# The is_cancerous_model is a sequential model with the following architecture:
#
# > **Conv2D layer:** A 2D convolutional layer with 32 filters, each of size 3x3, and a ReLU activation function. The input shape is (27, 27, 3), corresponding to the image dimensions and 3 channels (RGB). This layer has 896 trainable parameters.
#
# > **MaxPooling2D layer:** A 2D max pooling layer with a pool size of 2x2, which reduces the spatial dimensions of the input feature maps. There are no trainable parameters in this layer.
#
# > **Conv2D layer:** Another 2D convolutional layer with 64 filters, each of size 3x3, and a ReLU activation function. This layer has 18,496 trainable parameters.
#
# > **MaxPooling2D layer:** Another 2D max pooling layer with a pool size of 2x2, further reducing the spatial dimensions of the input feature maps. There are no trainable parameters in this layer.
#
# > **Conv2D layer:** A third 2D convolutional layer with 128 filters, each of size 3x3, and a ReLU activation function. This layer has 73,856 trainable parameters.
#
# > **MaxPooling2D layer:** A third 2D max pooling layer with a pool size of 2x2, further reducing the spatial dimensions of the input feature maps. There are no trainable parameters in this layer.
#
# > **Flatten layer:** This layer flattens the 2D feature maps into a 1D array with 128 elements, which can be used as input for the subsequent fully connected layers (Dense layers). There are no trainable parameters in this layer.
#
# > **Dense layer:** A fully connected layer with 64 neurons and a ReLU activation function. This layer has 8,256 trainable parameters.
#
# > **Dropout layer:** A dropout layer with a dropout rate of 0.5, which helps prevent overfitting by randomly setting 50% of the input units to 0 at each update during training. There are no trainable parameters in this layer.
#
# > **Dense layer:** The final fully connected layer with 1 neuron and a sigmoid activation function, used for binary classification. This layer has 65 trainable parameters.
#
# The total number of trainable parameters in the is_cancerous_model is 101,569.

# %%
# model architecture network summary for cell_type_model
cell_type_model.summary()

# %% [markdown]
# ### -------------> OBSERVATION:
#
# The cell_type_model is a sequential model with the following architecture:
#
# > **Conv2D layer:** A 2D convolutional layer with 32 filters, each of size 3x3, and a ReLU activation function. The input shape is (27, 27, 3), corresponding to the image dimensions and 3 channels (RGB). This layer has 896 trainable parameters.
#
# > **MaxPooling2D layer:** A 2D max pooling layer with a pool size of 2x2, which reduces the spatial dimensions of the input feature maps. There are no trainable parameters in this layer.
#
# > **Conv2D layer:** Another 2D convolutional layer with 64 filters, each of size 3x3, and a ReLU activation function. This layer has 18,496 trainable parameters.
#
# > **MaxPooling2D layer:** Another 2D max pooling layer with a pool size of 2x2, further reducing the spatial dimensions of the input feature maps. There are no trainable parameters in this layer.
#
# > **Conv2D layer:** A third 2D convolutional layer with 128 filters, each of size 3x3, and a ReLU activation function. This layer has 73,856 trainable parameters.
#
# > **MaxPooling2D layer:** A third 2D max pooling layer with a pool size of 2x2, further reducing the spatial dimensions of the input feature maps. There are no trainable parameters in this layer.
#
# > **Flatten layer:** This layer flattens the 2D feature maps into a 1D array with 128 elements, which can be used as input for the subsequent fully connected layers (Dense layers). There are no trainable parameters in this layer.
#
# > **Dense layer:** A fully connected layer with 64 neurons and a ReLU activation function. This layer has 8,256 trainable parameters.
#
# > **Dropout layer:** A dropout layer with a dropout rate of 0.5, which helps prevent overfitting by randomly setting 50% of the input units to 0 at each update during training. There are no trainable parameters in this layer.
#
# > **Dense layer:** The final fully connected layer with 4 neurons and a softmax activation function, used for multi-class classification. This layer has 260 trainable parameters.
#
# The total number of trainable parameters in the cell_type_model is 101,764.

# %% [markdown]
# <a id="9.2"></a>
# ## 9.2 Hyperparameter tuning CNN
#
# The motivation to progress from using a Convolutional Neural Network (CNN) model to hyperparameter tuning stems from the desire to improve the model's performance. Even though CNNs are generally good at handling image data, the initial model architecture and hyperparameters may not be optimal for the specific task. Therefore, hyperparameter tuning is performed to find the best possible model configuration.
#
# + **Optimal Model Configuration:** Every dataset is unique, and the best model configuration (architecture and hyperparameters) for one dataset may not necessarily be the best for another. Hyperparameter tuning allows us to find the best model configuration for the task and dataset.
#
# + **Improving Model Performance:** By tuning hyperparameters, we can often significantly improve the performance of our model (in terms of accuracy, precision, recall, F1 score, etc.). This is especially true when we do not know the best model configuration.
#
# + **Avoiding Overfitting and Underfitting:** Hyperparameter tuning can help us to find a good balance between bias and variance, thereby avoiding overfitting (high variance) and underfitting (high bias). For example, tuning the learning rate can help with convergence, and tuning the dropout rate can help with model regularization to prevent overfitting.
#
# + **Efficient Resource Use:** By finding the optimal hyperparameters, we can often train our model more efficiently, using fewer computational resources (CPU/GPU time, memory). This is especially important when dealing with large datasets and complex models.
#
# + **Better Understanding of the Model and Problem:** Finally, hyperparameter tuning can give us a better understanding of how different hyperparameters affect model performance, which can be very helpful for similar tasks in the future.
#
# We use Keras Tuner, a powerful and easy-to-use hyperparameter tuning library. It automatically finds the best hyperparameters for the model, saving us from the laborious and time-consuming process of manual tuning. Two key hyperparameters are tuned: **the number of units in the first Conv2D layer** and the **learning rate for the optimizer**. This should lead to a more accurate and efficient model.

# %%
% % time
# Define a function for model building for hyperparameter tuning


def build_is_cancerous_model(hp):
    model = Sequential()
    # Tune the number of units in the first Conv2D layer
    # Choose an optimal value between 32-128
    hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
    model.add(Conv2D(hp_units, (3, 3), activation='relu', input_shape=INPUT_DIM))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Instantiate the tuner and perform hypertuning
tuner = RandomSearch(
    build_is_cancerous_model,
    objective='val_accuracy',
    max_trials=5,  # number of different hyperparameter combinations to try
    # number of trials per hyperparameter combination to reduce results variance
    executions_per_trial=3,
    directory='my_dir',
    project_name='cancerous_tuning')

# Start the search for the best hyperparameters
tuner.search(data_gen.flow(train_images, train_is_cancerous_labels, batch_size=BATCH_SIZE),
             validation_data=(val_images, val_is_cancerous_labels),
             epochs=EPOCHS, verbose=2)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(
    f'Optimal number of units in the first Conv2D layer for build_is_cancerous_model: {best_hps.get("units")}')

# Build the model with the optimal hyperparameters
is_cancerous_model = tuner.hypermodel.build(best_hps)

# Same process can be repeated for the cell_type_model


def build_cell_type_model(hp):
    model = Sequential()
    # Tune the number of units in the first Conv2D layer
    # Choose an optimal value between 32-128
    hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
    model.add(Conv2D(hp_units, (3, 3), activation='relu', input_shape=INPUT_DIM))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(OUTPUT_CLASSES_CELLTYPE, activation='softmax'))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Instantiate the tuner and perform hypertuning
tuner = RandomSearch(
    build_cell_type_model,
    objective='val_accuracy',
    max_trials=5,  # number of different hyperparameter combinations to try
    # number of trials per hyperparameter combination to reduce results variance
    executions_per_trial=3,
    directory='my_dir',
    project_name='cell_type_tuning')

# Start the search for the best hyperparameters
tuner.search(data_gen.flow(train_images, train_cell_type_labels, batch_size=BATCH_SIZE),
             validation_data=(val_images, val_cell_type_labels),
             epochs=EPOCHS, verbose=2)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(
    f'Optimal number of units in the first Conv2D layer for build_cell_type_model: {best_hps.get("units")}')

# Build the model with the optimal hyperparameters
cell_type_model = tuner.hypermodel.build(best_hps)

# train the models
is_cancerous_history = is_cancerous_model.fit(data_gen.flow(train_images, train_is_cancerous_labels, batch_size=BATCH_SIZE),
                                              validation_data=(
                                                  val_images, val_is_cancerous_labels),
                                              epochs=EPOCHS, verbose=2)

cell_type_history = cell_type_model.fit(data_gen.flow(train_images, train_cell_type_labels, batch_size=BATCH_SIZE),
                                        validation_data=(
                                            val_images, val_cell_type_labels),
                                        epochs=EPOCHS, verbose=2)

# model architecture summary
is_cancerous_model.summary()
cell_type_model.summary()

# evaluate the models using the test set
is_cancerous_loss, is_cancerous_acc = is_cancerous_model.evaluate(
    test_images, test_is_cancerous_labels, verbose=2)
cell_type_loss, cell_type_acc = cell_type_model.evaluate(
    test_images, test_cell_type_labels, verbose=2)

print(
    f'Convolutional Neural Network after hyperparameter tuning model accuracy for is_cancerous prediction: {is_cancerous_acc}')
print(
    f'Convolutional Neural Network after hyperparameter tuning model accuracy for cell_type prediction: {cell_type_acc}')
print(
    f'Convolutional Neural Network after hyperparameter tuning model loss for is_cancerous prediction: {is_cancerous_loss}')
print(
    f'Convolutional Neural Network after hyperparameter tuning model loss for cell_type prediction: {cell_type_loss}')

# plot learning curves
plot_learning_curve(is_cancerous_history,
                    'is_cancerous_CNN_hyperparameter_tuning_model')
plot_learning_curve(cell_type_history,
                    'cell_type_CNN_hyperparameter_tuning_model')

# get predictions
is_cancerous_predictions = is_cancerous_model.predict(test_images)
# convert predictions to binary
is_cancerous_predictions = np.where(is_cancerous_predictions > 0.5, 1, 0)
cell_type_predictions = cell_type_model.predict(test_images)
# convert cell_type predictions to one-hot encoding
cell_type_predictions = np.argmax(cell_type_predictions, axis=1)

# get the confusion matrices
plot_confusion_matrix(test_is_cancerous_labels, is_cancerous_predictions,
                      'is_cancerous_CNN_hyperparameter_tuning_model')
plot_confusion_matrix(test_cell_type_labels.argmax(
    axis=1), cell_type_predictions, 'cell_type_CNN_hyperparameter_tuning_model')

# get the classification reports
print(
    f'Classification report for is_cancerous prediction:\n{classification_report(test_is_cancerous_labels, is_cancerous_predictions)}')
print(
    f'Classification report for cell_type prediction:\n{classification_report(test_cell_type_labels.argmax(axis=1), cell_type_predictions)}')

cnn_hyperparameter_tuning_model_df = evaluate_and_report(
    test_is_cancerous_labels,
    is_cancerous_predictions,
    test_cell_type_labels,
    cell_type_predictions,
    is_cancerous_model,
    cell_type_model,
    'CNN Hyper Tuning (Is Cancerous)',
    'CNN Hyper Tuning (Cell Type)')
# concatenate the cnn_hyperparameter_tuning_model_df to the evaluate_and_report_df
evaluate_and_report_df = pd.concat(
    [evaluate_and_report_df, cnn_hyperparameter_tuning_model_df], ignore_index=True)
evaluate_and_report_df

# %% [markdown]
# ### ------------> OBSERVATION:
#
# In our evaluation of the hyperparameter tuning of the Convolutional Neural Network (CNN) model, we observed some interesting dynamics in performance over the training period.
#
# #### Is Cancerous Model
# >> For the 'is_cancerous' prediction task, the model's performance on the validation dataset declined after 10 epochs. This is indicated by a decrease in accuracy and an increase in loss. Although the model achieved a high precision of 96.56% in predicting whether a cell is cancerous, the recall rate was relatively low at 41.71%. This discrepancy in precision and recall is likely due to the model **overfitting the training data**, resulting in poorer performance on the validation data. Despite this, the Area Under the Receiver Operating Characteristic (ROC AUC) score was still high at 0.93, indicating that the model can distinguish between cancerous and non-cancerous cells to a large extent. However, the declining performance on the validation set after 10 epochs suggests that further improvements could be made by reducing overfitting.
#
# #### Cell Type Model
# >> On the other hand, the 'cell_type' prediction task improved performance on the validation set compared to the training set after 10 epochs. The model achieved an accuracy of 73% on the validation set, with varying precision, recall, and F1-score levels across different cell types. The model performed well in identifying cell type 2 but struggled with cell type 3. The ROC AUC score for this task was 0.77, indicating good performance in differentiating between the various cell types.
#
# #### Recommendations to Improve Performance
# >> The CNN model's performance varies across tasks and datasets. For the `is_cancerous` prediction task, further work could be done to reduce overfitting, while for `cell_type` prediction, efforts could be made to improve the model's ability to identify certain cell types. Despite these areas for improvement, the model still demonstrated a solid ability to perform both tasks reasonably.

# %%
# save the model
is_cancerous_model.save(
    'models/is_cancerous_CNN_hyperparameter_tuning_model.h5')
cell_type_model.save('models/cell_type_CNN_hyperparameter_tuning_model.h5')
print('CNN Models after hyperparameter tuning saved successfully!')


# %% [markdown]
# <a id="9.3"></a>
# ## 9.3 Simple MLP Deep Learning Model
#
# The transition to a Simple Multi-Layer Perceptron (MLP) deep learning model from hyperparameter tuning is another step to explore the performance of different models on the same dataset. There are several reasons why one might consider this:
#
# + **Simplicity:** MLPs are generally simpler models than CNNs. They are easier to understand, implement, and require less computational resources. This can be beneficial in scenarios where the added complexity of a CNN does not lead to significantly improved performance.
# + **Feature Interaction:** MLPs, being fully connected networks, allow for interaction between all features in the data. In CNNs, the local receptive field of the convolution operation limits interaction between distant features. In some cases, MLPs may capture important global patterns in the data better than CNNs.
# + **Computational Efficiency:** MLPs are computationally efficient and require less training than complex models like CNNs. This can be advantageous in scenarios with limited resources or when quick prototyping is desired.
#
# In the provided script, two MLP models are created - one for predicting whether a cell is cancerous and another for predicting the cell type. The models consist of dense layers, with dropout layers in between for regularization. The models are trained on the same dataset used for the CNNs, allowing for a fair performance comparison. **The motivation here is to explore the potential of simpler, more computationally efficient models for the task.**
#
# Multilayer perceptron (MLP) is a type of neural network architecture, and it is indeed a form of deep learning model when it has more than one hidden layer. The architecture of the proposed Deep Learning model is designed to handle complex data patterns. Motivations behind choosing this architecture:
#
# + **Multi-Layer Design:** The model is a multi-layer perceptron (MLP) with three hidden layers. This design allows the model to learn complex patterns and relationships in the data. With only one layer, the model can only learn linear relationships. However, with multiple layers, the model can learn non-linear relationships often present in complex data like images.
# + **Number of Neurons:** The model starts with a larger number of neurons in the first layer (512) and gradually reduces the number of neurons in subsequent layers (256 and 128). This design is often referred to as a "funnel" structure. It is based on the assumption that the model should learn a large number of low-level features in the early layers and then combine these into a smaller number of high-level features in the deeper layers.
# + **Relu Activation Function:** The Rectified Linear Unit (ReLU) activation function is used in the hidden layers. ReLU is popular because it helps mitigate the vanishing gradients problem, which can hinder learning in deep neural networks.
# + **Dropout Layers:** Dropout layers are included after each Dense layer. Dropout is a regularization technique that helps prevent overfitting. During training, it randomly sets a fraction of the input units to 0, which helps prevent over-dependence on any single neuron and makes the model more robust.
# + **Sigmoid and Softmax Activation Functions:** In the output layer, a sigmoid activation function is used for the binary classification task (is_cancerous), and a softmax activation function is used for the multi-class classification task (cell_type). These activation functions are suitable for these types of classification tasks because they produce a probability distribution over the classes.
# + **Adam Optimizer:** The model uses the Adam optimizer, an adaptive learning rate optimization algorithm. It is popular because it requires little memory, is computationally efficient, and works well with problems that have large data or parameters.

# %%
% % time
# Define model architecture for "is_cancerous"


def create_deep_is_cancerous_model():
    model = Sequential(name='deep_is_cancerous_model')
    model.add(Flatten(input_shape=INPUT_DIM))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Define model architecture for "cell_type"


def create_deep_cell_type_model():
    model = Sequential(name='deep_cell_type_model')
    model.add(Flatten(input_shape=INPUT_DIM))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(OUTPUT_CLASSES_CELLTYPE, activation='softmax'))
    return model


# Instantiate models
deep_is_cancerous_model = create_deep_is_cancerous_model()
deep_cell_type_model = create_deep_cell_type_model()

# Compile models
deep_is_cancerous_model.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
deep_cell_type_model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train models
deep_is_cancerous_model.fit(train_images, train_is_cancerous_labels, epochs=EPOCHS,
                            batch_size=BATCH_SIZE, validation_data=(val_images, val_is_cancerous_labels))
deep_cell_type_model.fit(train_images, train_cell_type_labels, epochs=EPOCHS,
                         batch_size=BATCH_SIZE, validation_data=(val_images, val_cell_type_labels))

# plot learning curves
plot_learning_curve(deep_is_cancerous_model.history,
                    'is_cancerous_deep_learning_model')
plot_learning_curve(deep_cell_type_model.history,
                    'cell_type_deep_learning_model')

# get predictions
deep_is_cancerous_predictions = deep_is_cancerous_model.predict(test_images)
# convert predictions to binary
deep_is_cancerous_predictions = np.where(
    deep_is_cancerous_predictions > 0.5, 1, 0)
deep_cell_type_predictions = deep_cell_type_model.predict(test_images)
# convert cell_type predictions to one-hot encoding
deep_cell_type_predictions = np.argmax(deep_cell_type_predictions, axis=1)

# get the confusion matrices
plot_confusion_matrix(test_is_cancerous_labels,
                      deep_is_cancerous_predictions, 'is_cancerous_deep_learning_model')
plot_confusion_matrix(test_cell_type_labels.argmax(
    axis=1), deep_cell_type_predictions, 'cell_type_deep_learning_model')

# get the classification reports
print(
    f'Classification report for is_cancerous prediction:\n{classification_report(test_is_cancerous_labels, deep_is_cancerous_predictions)}')
print(
    f'Classification report for cell_type prediction:\n{classification_report(test_cell_type_labels.argmax(axis=1), deep_cell_type_predictions)}')

deep_learning_model_df = evaluate_and_report(
    test_is_cancerous_labels,
    deep_is_cancerous_predictions,
    test_cell_type_labels,
    deep_cell_type_predictions,
    deep_is_cancerous_model,
    deep_cell_type_model,
    'Deep Learning (Is Cancerous)',
    'Deep Learning (Cell Type)')
# concatenate the deep_learning_model_df to the evaluate_and_report_df
evaluate_and_report_df = pd.concat(
    [evaluate_and_report_df, deep_learning_model_df], ignore_index=True)

# save the model
deep_is_cancerous_model.save('models/is_cancerous_deep_learning_model.h5')
deep_cell_type_model.save('models/cell_type_deep_learning_model.h5')
print('Deep Learning Models saved successfully!')

evaluate_and_report_df


# %% [markdown]
# ### --------> OBSERVATION:
#
# #### Is Cancerous:
# >> the MLP model exhibited an interesting phenomenon where the validation loss and accuracy were better than the training datasets. This could indicate that the model generalizes well from the training data to unseen data. On the test set, the model achieved an accuracy of 81% with a good macro F1-score of 80%. Interestingly, the model had a high recall rate of 94.31% but a lower precision of 69.46%, suggesting that it was effective at identifying cancerous cells but also produced many false positives. The Area Under the Receiver Operating Characteristic (ROC AUC) score was 0.83, indicating a good level of discriminative ability between cancerous and non-cancerous cells.
#
# #### Cell Type:
#
# >> However, the model's performance on the 'cell_type' prediction task differed. Here, the model's validation loss and accuracy were worse than the training dataset's, indicating potential overfitting on the training data. The performance on the test set was subpar, with an accuracy of just 41% and a very low macro F1-score of 14.49%. The precision, recall, and F1-scores for cell types 0, 1, and 3 were all 0, suggesting that the model completely failed to identify these cell types. It only managed to identify cell type 2, leading to a precision of 41% for that type. The ROC AUC score was exactly 0.5, indicating that the model's ability to distinguish between the different cell types was no better than random chance.
#
# > In summary, the Simple MLP model showed promising results for the `is_cancerous` prediction task, with good generalization from the training to the validation set and reasonable performance on the test set. However, it performed poorly on the `cell_type` prediction task, suggesting that this model may not be suitable for multiclass classification problems or require further tuning or a different architecture to handle such tasks effectively.

# %%
# export the evaluate_and_report_df to csv
evaluate_and_report_df.to_csv(
    '../output_data/evaluate_and_report.csv', index=False)
print('evaluate_and_report_df exported into CSV successfully!')

# %% [markdown]
# ### Choosing models
#
# The "best" model depends on what we value most in your specific context. However, we generally look at a combination of precision, recall, f1-score, and Area Under the Curve (AUC) to make a judgment.
#
# + If we care most about overall accuracy, look at the f1-score, the harmonic mean of precision and recall. The higher the f1-score, the better the model is at accurately classifying positive instances.
# + If we care most about minimizing false positives, then look at precision. The higher the precision, the fewer false positives a model makes.
# + If we care most about minimizing false negatives, consider a recall. The higher the recall, the fewer false negatives a model makes.
# + If we care about the model's rank ordering of predictions, look at AUC. The higher the AUC, the better the model is at ranking positive instances above negative ones.
#
# > Looking at our `evaluate_and_report_df` table, for the `Is Cancerous` category, the **"CNN Hyper Tuning (Is Cancerous)"** model has the highest precision, but its recall is quite low. This indicates that while it is very confident in its positive predictions, it is likely missing a lot of positive instances. On the other hand, the **"Deep Learning (Is Cancerous)"** model has lower precision but very high recall, indicating it is good at finding positive instances but may have a higher rate of false positives. It also has the highest f1-score and AUC in this category, so that it might be the best overall model for this task.
# For the `Cell Type` category, the **"CNN Hyper Tuning (Cell Type)"** model has the highest f1-score and AUC, so that it might be the best model for this task. Therefore we will use CNN model to predict the extra data

# %% [markdown]
# <a id="10"></a>
# # 10. Extra Data

# %%
# cellType values for each cellTypeName
print(
    f'\nThe cellType values for each cellTypeName in the main data: \n{main_data.groupby(["cellTypeName"])["cellType"].unique().reset_index(name="cellType")}')

# %% [markdown]
# <a id="10.1"></a>
# ## 10.1 Extra data Cell Type Predictions

# %%
# load CNN hyperparameter tuned models
cell_type_model = load_model(
    'models/cell_type_CNN_hyperparameter_tuning_model.h5')
# Predict cell types for extra data using the cell-type classification model
extra_cell_type_pred = cell_type_model.predict(extra_images)
extra_cell_type_pred
# convert cell_type predictions to one-hot encoding
extra_cell_type_pred = np.argmax(extra_cell_type_pred, axis=1)
# add extra_cell_type_pred to extra_data as cellType column
extra_data['cellType'] = extra_cell_type_pred
# add cellTypeName column to extra_data
extra_data['cellTypeName'] = extra_data['cellType'].map(
    {0: 'fibroblast', 1: 'inflammatory', 2: 'epithelial', 3: 'others'})
# export extra_data to csv in the output_data folder
extra_data.to_csv('../output_data/extra_data.csv', index=False)
# print the first 3 rows of extra_data
extra_data.head(3)

# %% [markdown]
# <a id="10.2"></a>
# ## 10.2 Retrain model on the combined data
#
# The motivation to further train the existing CNN models, which were optimized through hyperparameter tuning, on a combined dataset stems from the following compelling reasons:
#
# + **Data Augmentation:** The amalgamation of two datasets augments the data quantity available for model training. This data enrichment typically leads to enhanced model performance, enabling the model to learn more intricate and nuanced data relationships.
#
# + **Enhanced Model Generalization:** The additional and diverse data from the combined set mitigates the likelihood of model overfitting to specific patterns in a single dataset, leading to superior generalization. Consequently, the model's performance on unseen data is likely to improve.
#
# + **Robustness Boost:** Using a combined dataset can bolster the model's robustness. This is attributed to the increased chances of the model encountering and learning from outliers or atypical data points.
#
# + **Performance Uplift:** Hyperparameter tuning is a strategy to identify the optimal model architecture and parameters for a given task. By harnessing the combined datasets, the model is endowed with a broader context for the task, potentially augmenting its performance.
#
# + **Real-world Compatibility:** The amalgamation of datasets could better mirror real-world data's intricacy and variability, culminating in more practically useful models.

# %%
% % time
# combine main_data and extra_data
combined_data = pd.concat([main_data, extra_data], ignore_index=True)
# export combined_data to csv in the output_data folder
combined_data.to_csv('../output_data/combined_data.csv', index=False)

# load the is_cancerous_CNN_hyperparameter_tuning_model
is_cancerous_model = load_model(
    'models/is_cancerous_CNN_hyperparameter_tuning_model.h5')

# preprocess the images in the combined dataset
combined_images = load_and_preprocess_images(combined_data)
# One-hot encode the cellType
encoder = OneHotEncoder(sparse=False)
combined_cell_type_labels = encoder.fit_transform(
    combined_data['cellType'].values.reshape(-1, 1))
# not one-hot encoded for binary classification
combined_is_cancerous_labels = combined_data['isCancerous'].values.reshape(
    -1, 1)

# train the CNN hyperparameter tuned model on the combined data
cell_type_model.fit(combined_images, combined_cell_type_labels, epochs=EPOCHS,
                    batch_size=BATCH_SIZE, validation_data=(val_images, val_cell_type_labels))
is_cancerous_model.fit(combined_images, combined_is_cancerous_labels, epochs=EPOCHS,
                       batch_size=BATCH_SIZE, validation_data=(val_images, val_is_cancerous_labels))

# plot learning curves
plot_learning_curve(is_cancerous_model.history,
                    'is_cancerous_CNN_hyperparameter_tuning_model on combined data')
plot_learning_curve(cell_type_model.history,
                    'cell_type_CNN_hyperparameter_tuning_model on combined data')

# get predictions
is_cancerous_predictions = is_cancerous_model.predict(test_images)
# convert predictions to binary
is_cancerous_predictions = np.where(is_cancerous_predictions > 0.5, 1, 0)
cell_type_predictions = cell_type_model.predict(test_images)
# convert cell_type predictions to one-hot encoding
cell_type_predictions = np.argmax(cell_type_predictions, axis=1)

# get the confusion matrices
plot_confusion_matrix(test_is_cancerous_labels, is_cancerous_predictions,
                      'is_cancerous_CNN_hyperparameter_tuning_model')
plot_confusion_matrix(test_cell_type_labels.argmax(
    axis=1), cell_type_predictions, 'cell_type_CNN_hyperparameter_tuning_model')

# get the classification reports
print(
    f'Classification report for is_cancerous prediction:\n{classification_report(test_is_cancerous_labels, is_cancerous_predictions)}')
print(
    f'Classification report for cell_type prediction:\n{classification_report(test_cell_type_labels.argmax(axis=1), cell_type_predictions)}')

# save CNN hyperparameter tuned models on combined data
is_cancerous_model.save(
    'models/is_cancerous_CNN_hyperparameter_tuning_model_on_combined_data.h5')
cell_type_model.save(
    'models/cell_type_CNN_hyperparameter_tuning_model_on_combined_data.h5')
print('Models saved successfully!')

# evaluate the model
evaluate_and_report_combined_data_df = evaluate_and_report(
    test_is_cancerous_labels,
    is_cancerous_predictions,
    test_cell_type_labels,
    cell_type_predictions,
    is_cancerous_model,
    cell_type_model,
    'CNN Hyperparameter Tuning on combined data (Is Cancerous)',
    'CNN Hyperparameter Tuning on combined data (Cell Type)')

# concatenate the evaluate_and_report_combined_data_df to the evaluate_and_report_df
evaluate_and_report_df = pd.concat(
    [evaluate_and_report_df, evaluate_and_report_combined_data_df], ignore_index=True)
evaluate_and_report_df

# %% [markdown]
# ### --------> OBSERVATION:
#
# The Convolutional Neural Network (CNN) model, after hyperparameter tuning and retraining on the combined dataset, demonstrates varied performance for the 'is_cancerous' and 'cell_type' prediction tasks.
#
# #### Is Cancerous:
# >> In the 'is_cancerous' prediction, the validation's loss and accuracy were superior to the training set's results. This implies that the model generalized well from the training data to unseen data. With a high accuracy of 89%, the model's performance on the test set was commendable. The model balanced precision (88.86%) and recall (84.9%) fairly well, resulting in a high macro F1-score of 86.84%. The ROC AUC score of 0.89 suggests the model could differentiate between cancerous and non-cancerous cells.
#
# #### Cell Type:
# >> Contrarily, for the 'cell_type' prediction, the validation's loss and accuracy were worse than the training set's results, hinting at potential overfitting to the training data. Despite this, the model still performed reasonably on the test set with an accuracy of 77%. The precision, recall, and F1-score varied significantly among the different cell types. Notably, for cell type 3, the model had a high precision of 83% but a low recall of 26%, leading to a relatively low macro F1-score of 40%. This suggests that the model struggled to identify this cell type correctly. The overall ROC AUC score of 0.80 indicates that the model's ability to distinguish between different cell types was good but less robust than the 'is_cancerous' prediction.
#
# > In summary, the CNN hyperparameter tuning model retrained on the combined data performed excellently on the 'is_cancerous' prediction task and reasonably on the 'cell_type' prediction task. The performance differences suggest that further optimization or a more complex model might be necessary for multiclass cell type classification. The model training and evaluation process was relatively time-efficient, taking just over 3 minutes of wall time.

# %%
# export evaluate_and_report_df to csv in the output_data folder
evaluate_and_report_df.to_csv(
    '../output_data/evaluate_and_report_df.csv', index=False)
print('evaluate_and_report_df saved successfully!')

# %%
# load ../output_data/evaluate_and_report_df.csv data
evaluate_and_report_df = pd.read_csv(
    '../output_data/evaluate_and_report_df.csv')
# delete model_params column
del evaluate_and_report_df['model_params']
# save evaluate_and_report_df_without_model_params.csv to output_data folder
evaluate_and_report_df.to_csv(
    '../output_data/evaluate_and_report_df_without_model_params.csv', index=False)
evaluate_and_report_df

# %% [markdown]
# <a id="11"></a>
# # 11. Best model
#
# This report compares several models to predict whether cells are cancerous (binary classification) and to identify the cell type (multiclass classification). The models include Traditional Neural Network, Convolutional Neural Network (CNN), CNN with Hyperparameter Tuning, and a Deep Learning model using Multilayer Perceptron (MLP). The performance metrics include precision, recall, F1-score, and the area under the receiver operating characteristic curve (AUC).
#
# + For the binary classification task of predicting 'is_cancerous':
# The CNN model with Hyperparameter Tuning trained on combined data achieved the best performance with a precision of 88.86%, recall of 84.9%, F1-score of 86.84%, and AUC of 0.89.
#
# + For the multiclass classification task of predicting 'cell_type':
# The CNN model with Hyperparameter Tuning trained on combined data also came out on top, achieving a precision of 77.63%, recall of 68.35%, F1-score of 68.83%, and AUC of 0.80.
#
# > The Traditional Neural Network and Deep Learning models, particularly for the 'cell_type' prediction task, performed poorly compared to the CNN models. Their macro F1-scores and AUC scores were significantly lower.
#
# > Based on the above evaluation metrics and considering the balance between precision, recall, F1-score, and AUC, the Convolutional Neural Network model with Hyperparameter Tuning trained on the combined data is the best model for both tasks in this study. **However, upon closer inspection, we observed that this model tended to overfit the training data, which could lead to poor generalization when applied to unseen data.**
#
# + After considering all factors, we have decided to proceed with the CNN model with Hyperparameter Tuning trained on the main data for the cell-type classification task. This decision was made because this model showed a better balance between performance on the training data and generalization to new data. The model did not overfit the training data, and thus, it is expected to have better generalization capabilities, making it more suitable for real-world applications.
#
# + It is important to note that model selection is not solely based on performance metrics. The model's ability to generalize to unseen data is a crucial aspect to consider. While the model trained on the combined data had slightly higher metrics, the risk of overfitting makes the CNN model with Hyperparameter Tuning trained on the main data a more reliable choice for future predictions.
#
# + We will use the CNN model with Hyperparameter Tuning trained on the main data for our cancer cell-type prediction tasks and compare it to other's research papers' performance using the same data.

# %% [markdown]
# <a id="12"></a>
# # 12. Independent Evaluation
# Compare performance to other works in literature that use similar data

# %% [markdown]
# ## There are 3 existing papers that will be compared to our work:
#
# ### **1. An Efficient Method of Histological Cell Image Detection Based on Spatial Information Convolution Neural Network - CNN(I,C,M,F,S1,S2)**
#
# This study use Convolution Neural Network consist of:
# - input(I),
# - convolution(C),
# - max-pooling(M),
# - fully-connected(F),
# - parameter estimation(S1)
# - spatial layers(S2).
#
# Their proposed model's performance: ***"cellType": {"accuracy": 0.758, "precision": 0.781, "recall": 0.823, "f1_score": 0.802}***
#
# ### **2. Semantic Segmentation of Cell Nuclei in Breast Cancer using Convolutional Neural Network - CNN**
#
# In this study, they proposed Convolutional Neural Network, it has three main layers where each main layer contains three sublayers, which will be the input layer and a total of ten layers:
# - First main layer contains convolution layer and rectified linear unit and maxpool
# - Second main layer contains convolutional layer rectified unit and transposed convolution layer (the non-convoluted layer)
# - The deconvolution layer reverses the operation of the standard convolution layer, restoring the original input
# - Last main layer contains convolutional layer, softmax and classification layer
#
# Their proposed model's performance: ***"cellType": {"accuracy": 0.851, "precision": 0.844, "recall": 0.832, "f1_score": None}***
#
# ### **3. Locality Sensitive Deep Learning for Detection and Classification of Nuclei in Routine Colon Cancer Histology Images - softmax CNN + NEP**
#
# This paper introduces a new approach called Neighboring Ensemble Predictor (NEP) is proposed, which combines CNN with spatial ensembling for nucleus classification. NEP utilizes predictions from relevant patches in the local vicinity of the nucleus to be classified, resulting in more precise classification outcomes compared to the traditional single-patch based method.
#
# Their proposed model's performance: ***"cellType": {"f1_score": 0.784, "auc": 0.917}***

# %%
# extract CNN Hyperparameter Tuning on combined data (Cell Type) current model performance from the evaluate_and_report_df
evaluate_and_report_df = pd.read_csv(
    '../output_data/evaluate_and_report_df.csv')
evaluate_and_report_df = evaluate_and_report_df[evaluate_and_report_df['model_name']
                                                == 'CNN Hyper Tuning (Cell Type)']
evaluate_and_report_df

# extract the current model performance
current_model_performance = evaluate_and_report_df.iloc[0, :-2].to_dict()
current_model_performance

# %%
# extract CNN Hyperparameter Tuning on combined data (Cell Type) current model performance from the evaluate_and_report_df
evaluate_and_report_df = pd.read_csv(
    '../output_data/evaluate_and_report_df.csv')
evaluate_and_report_df = evaluate_and_report_df[evaluate_and_report_df['model_name']
                                                == 'CNN Hyperparameter Tuning on combined data (Cell Type)']

# extract the current model performance
current_model_performance = evaluate_and_report_df.iloc[0, 2:].to_dict()

my_model_performance = {
    "CNN Hyperparameter Tuning on combined data (Cell Type)":
    {'precision': 0.6675656171899069,
     'recall': 0.6342946365944319,
     'f1-score': 0.6318615211270934,
     'auc': 0.770051144427877}
}

# existing works performance
existing_works = {
    "An Efficient Method of Histological Cell Image Detection Based on Spatial Information Convolution Neural Network - CNN(I,C,M,F,S1,S2) ": {
        "accuracy": 0.758, "precision": 0.781, "recall": 0.823, "f1_score": 0.802
    },
    "Semantic Segmentation of Cell Nuclei in Breast Cancer using Convolutional Neural Network - CNN ": {
        "accuracy": 0.851, "precision": 0.844, "recall": 0.832, "f1_score": None
    },
    "Locality Sensitive Deep Learning for Detection and Classification of Nuclei in Routine Colon Cancer Histology Images - softmax CNN + NEP": {
        "f1_score": 0.784, "auc": 0.917
    }
}

# create a DataFrame for easier manipulation
df_my_model = pd.DataFrame(my_model_performance).T
df_existing_works = pd.DataFrame(existing_works).T

# concatenate your model performance with the existing works
df_compare = pd.concat([df_my_model, df_existing_works],
                       keys=['My Model', 'Existing Works'])

df_compare

# %% [markdown]
# ### -------------> OBSERVATION:
#
# > Based on the table, the "Locality Sensitive Deep Learning for Detection and Classification of Nuclei in Routine Colon Cancer Histology Images - softmax CNN + NEP" model performs best on the AUC and F1-score metrics but lacks values for precision, recall, and accuracy. While having a lower AUC value, our model provides a complete evaluation, including precision and recall.
#
# > It would be ideal to compare these models better if they were evaluated on the same metrics. Moreover, it is essential to consider the problem's specifics - in some cases, certain metrics might be more important than others. For instance, in a medical diagnosis scenario, a **high recall** might be more desirable than high precision to **minimize false negatives**.
#
# > Recall: Our model has a recall of 0.634295. This is lower than the recall of the "Spatial Information Convolution Neural Network" model (0.823) and the "Semantic Segmentation" model (0.832).

# %%
# # clean up: terminate the kernel and free memory resources
# import os, signal
# os.kill(os.getpid(), signal.SIGKILL)

# %% [markdown]
# <a id="13"></a>
# # 13. References
#
# + https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
# + https://www.tensorflow.org/api_docs/python/tf/keras/layers/
# + https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# + https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# + https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
# + https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam
# + https://dl-acm-org.ezproxy.lib.rmit.edu.au/doi/pdf/10.1145/3376067.3376109
# + https://www.ijcaonline.org/archives/volume176/number22/zarbega-2020-ijca-920133.pdf
# + https://ieeexplore-ieee-org.ezproxy.lib.rmit.edu.au/stamp/stamp.jsp?tp=&arnumber=7399414
#
#
# #### [Back to Table of Contents](#0)
