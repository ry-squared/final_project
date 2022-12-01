import pandas as pd
import os
import shutil
import random
import warnings
import zipfile
import shutil
warnings.simplefilter(action='ignore', category=FutureWarning)


covid_normal_zip_dir = '/Users/ryanwest/OMSCS/cs6440/final-project/data/'
shutil.rmtree(covid_normal_zip_dir + "Dataset")
with zipfile.ZipFile(covid_normal_zip_dir + "archive.zip", 'r') as zip_ref:
    zip_ref.extractall(covid_normal_zip_dir)
pred_dir_images = os.listdir(covid_normal_zip_dir + 'Dataset/Prediction/')

os.mkdir(covid_normal_zip_dir + 'Dataset/Prediction/Covid/')
os.mkdir(covid_normal_zip_dir + 'Dataset/Prediction/Normal/')
# os.mkdir(covid_normal_zip_dir + 'Dataset/Prediction/pneumonia/')
# os.mkdir(covid_normal_zip_dir + 'Dataset/Train/pneumonia/')
# os.mkdir(covid_normal_zip_dir + 'Dataset/Val/pneumonia/')

for image in pred_dir_images:
    if "NORMAL" in image:
        shutil.move(covid_normal_zip_dir + 'Dataset/Prediction/' + image, covid_normal_zip_dir + 'Dataset/Prediction/Normal/' + image )
    elif "pneumonia" not in image:
        shutil.move(covid_normal_zip_dir + 'Dataset/Prediction/' + image, covid_normal_zip_dir + 'Dataset/Prediction/Covid/' + image)


# unzip NIH archive
NIH_archive = "/Users/ryanwest/OMSCS/cs6440/final-project/data/NIH_dataset/"
if not os.path.exists(NIH_archive + "archive"):
    with zipfile.ZipFile(NIH_archive + "archive.zip", 'r') as zip_ref:
        zip_ref.extractall(NIH_archive + "archive")

NIH_dir = "/Users/ryanwest/OMSCS/cs6440/final-project/data/NIH_dataset/archive/"
NIH_data_labels = NIH_dir + "sample_labels.csv"
NIH_data_images = NIH_dir + "sample/sample/images/"

dest_dir = "/Users/ryanwest/OMSCS/cs6440/final-project/data/Dataset/"

labels = pd.read_csv(NIH_data_labels)
labels["Finding Labels lowercase"] = labels["Finding Labels"].str.lower()

pneumonia_file_names = labels[labels["Finding Labels lowercase"].str.contains("pneumonia")]["Image Index"]

random.seed(123)
random_nums = [random.random() for num in list(range(len(pneumonia_file_names)))]

if not os.path.exists(dest_dir + "Train/pneumonia/"):
    os.mkdir(dest_dir + "Train/pneumonia/")

if not os.path.exists(dest_dir + "Val/pneumonia/"):
    os.mkdir(dest_dir + "Val/pneumonia/")

if not os.path.exists(dest_dir + "Prediction/pneumonia/"):
    os.mkdir(dest_dir + "Prediction/pneumonia/")

for image_idx, image_name in enumerate(pneumonia_file_names):
    random_num = random_nums[image_idx]
    if random_num < 0.7:
        shutil.copy(NIH_data_images + image_name, dest_dir + "Train/pneumonia/" + "pneumonia_" + image_name )
    elif 0.7 <= random_num < 0.85:
        shutil.copy(NIH_data_images + image_name, dest_dir + "Val/pneumonia/" + "pneumonia_" + image_name )
    elif random_num >= 0.85:
        shutil.copy(NIH_data_images + image_name, dest_dir + "Prediction/pneumonia/" + "pneumonia_" + image_name )



# meta_df = pd.read_csv("/Users/ryanwest/Downloads/metadata.csv")
#
# meta_df["finding_lowercase"] = meta_df.finding.str.lower()
# meta_df[(meta_df.finding_lowercase.str.contains("covid")) & ~(meta_df.finding_lowercase.str.contains("pneu"))]
# meta_df.finding_lowercase.unique()