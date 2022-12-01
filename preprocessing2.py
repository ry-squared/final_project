import pandas as pd
import os
import random
import warnings
import shutil
warnings.simplefilter(action='ignore', category=FutureWarning)


data_dir = '/Users/ryanwest/OMSCS/cs6440/final-project/data/'
shutil.rmtree(data_dir + "Dataset2")

covid_source_img_path = "/Users/ryanwest/OMSCS/cs6440/covid-chestxray-dataset/"
meta_df = pd.read_csv(covid_source_img_path + "metadata.csv")
covid_filenames = meta_df[meta_df.finding.str.contains("COVID-19")].filename.tolist()
covid_filenames = [image for image in os.listdir(covid_source_img_path + 'images') if image in covid_filenames]

normal_source_img_path = "/Users/ryanwest/OMSCS/cs6440/final-project/data/Dataset2/chest_xray/train/NORMAL/"
normal_filenames = os.listdir(normal_source_img_path)

pneumonia_source_img_path = "/Users/ryanwest/OMSCS/cs6440/final-project/data/Dataset2/chest_xray/train/PNEUMONIA/"
pneumonia_virus_filenames = [img for img in os.listdir(pneumonia_source_img_path) if "virus" in img]
pneumonia_bacteria_filenames = [img for img in os.listdir(pneumonia_source_img_path) if "bacteria" in img]

distribution = {"covid":len(covid_filenames),
                "normal":len(normal_filenames),
                "pneumonia_virus":len(pneumonia_virus_filenames),
                "bacteria_virus_filenames":len(pneumonia_bacteria_filenames)}

min_class_number = sorted(distribution.items(), key=lambda item: item[1])[0]

random.seed(123)

covid_filenames = random.sample(covid_filenames, min_class_number[1])
normal_filenames = random.sample(normal_filenames, min_class_number[1])
pneumonia_virus_filenames = random.sample(pneumonia_virus_filenames, min_class_number[1])
pneumonia_bacteria_filenames = random.sample(pneumonia_bacteria_filenames, min_class_number[1])

dest_dir = "/Users/ryanwest/OMSCS/cs6440/final-project/data/Dataset2/"

if not os.path.exists(dest_dir + "Train/"):
    os.mkdir(dest_dir + "Train/")

if not os.path.exists(dest_dir + "Val/"):
    os.mkdir(dest_dir + "Val/")

if not os.path.exists(dest_dir + "Prediction/"):
    os.mkdir(dest_dir + "Prediction/")

if not os.path.exists(dest_dir + "Train/covid/"):
    os.mkdir(dest_dir + "Train/covid/")

if not os.path.exists(dest_dir + "Val/covid/"):
    os.mkdir(dest_dir + "Val/covid/")

if not os.path.exists(dest_dir + "Prediction/covid/"):
    os.mkdir(dest_dir + "Prediction/covid/")

if not os.path.exists(dest_dir + "Train/normal/"):
    os.mkdir(dest_dir + "Train/normal/")

if not os.path.exists(dest_dir + "Val/normal/"):
    os.mkdir(dest_dir + "Val/normal/")

if not os.path.exists(dest_dir + "Prediction/normal/"):
    os.mkdir(dest_dir + "Prediction/normal/")

if not os.path.exists(dest_dir + "Train/pneumonia_virus/"):
    os.mkdir(dest_dir + "Train/pneumonia_virus/")

if not os.path.exists(dest_dir + "Val/pneumonia_virus/"):
    os.mkdir(dest_dir + "Val/pneumonia_virus/")

if not os.path.exists(dest_dir + "Prediction/pneumonia_virus/"):
    os.mkdir(dest_dir + "Prediction/pneumonia_virus/")

if not os.path.exists(dest_dir + "Train/pneumonia_bacteria/"):
    os.mkdir(dest_dir + "Train/pneumonia_bacteria/")

if not os.path.exists(dest_dir + "Val/pneumonia_bacteria/"):
    os.mkdir(dest_dir + "Val/pneumonia_bacteria/")

if not os.path.exists(dest_dir + "Prediction/pneumonia_bacteria/"):
    os.mkdir(dest_dir + "Prediction/pneumonia_bacteria/")

random.seed(123)
random_nums = [random.random() for num in list(range(len(covid_filenames)))]
for image_idx in range(len(covid_filenames)):

    random_num = random_nums[image_idx]
    covid_filename = covid_filenames[image_idx]
    normal_filename = normal_filenames[image_idx]
    pneumonia_virus_filename = pneumonia_virus_filenames[image_idx]
    pneumonia_bacteria_filename = pneumonia_bacteria_filenames[image_idx]

    if random_num < 0.7:
        shutil.copy(covid_source_img_path + "images/" + covid_filename, dest_dir + "Train/covid/" + covid_filename)
        shutil.copy(normal_source_img_path + normal_filename, dest_dir + "Train/normal/" + normal_filename)
        shutil.copy(pneumonia_source_img_path + pneumonia_virus_filename, dest_dir + "Train/pneumonia_virus/" + pneumonia_virus_filename)
        shutil.copy(pneumonia_source_img_path +  pneumonia_bacteria_filename, dest_dir + "Train/pneumonia_bacteria/" + pneumonia_bacteria_filename)
    elif 0.7 <= random_num < 0.85:
        shutil.copy(covid_source_img_path + "images/" + covid_filename, dest_dir + "Val/covid/" + covid_filename)
        shutil.copy(normal_source_img_path + normal_filename, dest_dir + "Val/normal/" + normal_filename)
        shutil.copy(pneumonia_source_img_path + pneumonia_virus_filename, dest_dir + "Val/pneumonia_virus/" + pneumonia_virus_filename)
        shutil.copy(pneumonia_source_img_path + pneumonia_bacteria_filename, dest_dir + "Val/pneumonia_bacteria/" + pneumonia_bacteria_filename)
    elif random_num >= 0.85:
        shutil.copy(covid_source_img_path + "images/" + covid_filename, dest_dir + "Prediction/covid/" + covid_filename)
        shutil.copy(normal_source_img_path + normal_filename, dest_dir + "Prediction/normal/" + normal_filename)
        shutil.copy(pneumonia_source_img_path + pneumonia_virus_filename, dest_dir + "Prediction/pneumonia_virus/" + pneumonia_virus_filename)
        shutil.copy(pneumonia_source_img_path + pneumonia_bacteria_filename, dest_dir + "Prediction/pneumonia_bacteria/" + pneumonia_bacteria_filename)
