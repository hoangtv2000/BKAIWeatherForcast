import json
import os

import pandas
import regex as re
# from keras.callbacks import TensorBoard
# from keras.preprocessing import sequence

import pickle
import time
from sklearn.naive_bayes import MultinomialNB

from keras.models import Sequential
from sklearn.model_selection import train_test_split
from underthesea import word_tokenize
from sklearn.utils import class_weight
import config
import utils


def load_csv():
    file_film = pandas.read_csv(os.path.join(config.folder_data, config.file_csv_247film), na_values="nan",
                                keep_default_na=False)
    dict_film = {}
    list_title = []
    list_description = []
    list_categories_2 = []

    for index, row in file_film.iterrows():
        # print(row)
        title = row['title']
        if row['director'] == "Đạo diễn:  Đang cập nhật":
            row['director'] = ""
        if row['country'].strip() == "Quốc gia:":
            row['country'] = ""
        if row['actor'].strip() == "['Đang cập nhật']" or row['actor'].strip() == "['']":
            row['actor'] = ""
        else:
            row["actor"] = "Diễn viên " + row['actor']
        description = ""
        if row['description'] != "nan":
            try:
                description = title + "\n" + row['description'] + "\n" + row['director'] + \
                              "\n" + row['country'] + "\n" + row['actor']

            except Exception as e:
                print(e)
                print("hello")
                print(title)
                print(row['description'])
                print(row['director'])
                print(row['country'])
                print(row['actor'])
                exit()
        else:
            print("hi")
            print(title)
            exit()

        description = text_preprocess(description)
        categories = " ".join(row['categories'].split()).lower().strip().replace('[', '').replace(']', ''). \
            replace("'", '').split(",")

        for i in range(0, len(categories)):
            categories[i] = categories[i].strip()
            categories[i] = text_preprocess(categories[i], type="categories")

        # vì toàn bộ các nhãn của 247 phim đều có categories là film mới nên ta sẽ loại bỏ nó.
        categories.remove("phim mới")
        # loại bỏ nhãn phim không có ý nghĩa
        if "không_thể bỏ lỡ" in categories:
            categories.remove("không_thể bỏ lỡ")

        if row['description'] != "nan" and description != "" and len(categories) != 0:
            if title not in dict_film:
                dict_film[title] = {
                    "description": description,
                    "categories": categories
                }

            else:
                dict_film.get(title)["description"] = dict_film.get(title).get("description") + description
                list_categories = dict_film.get(title).get("categories")
                for category in categories:
                    if category not in list_categories:
                        list_categories.append(category)
                dict_film.get(title)["categories"] = list_categories
            list_title.append(title)
            list_description.append(description)
            list_categories_2.append(categories)

    print(len(list_title))
    print(len(list_description))
    print(len(list_categories_2))
    # exit()
    dict_film2 = {"title": list_title, "description": list_description, "categories": list_categories_2}
    new_data_film = pandas.DataFrame(data=dict_film2)
    with open(os.path.join(config.folder_data, "data_247film.txt"), "w") as outfile:
        outfile.write(json.dumps(dict_film, indent=4))
    # print(dict_film)
    new_data_film.to_csv("data.csv", index=False)
    return dict_film


def text_preprocess(document, type="categories"):
    document = utils.chuan_hoa_dau_cau_tieng_viet(document)
    document = document.lower()
    document = re.sub(r"[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]", ' ', document)
    document = re.sub(r'\s+', ' ', document).strip()
    if type == "categories":
        document = word_tokenize(document, format="text")
    else:
        document = word_tokenize(document)
    return document


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dict_film = load_csv()
    # print(text_preprocess(dict_film.get("The Pentaverate (Season 1) (2022)").get("description")))
    count_category = {}
    # dict_categories = {}
    vocab = {}
    word_description_film = {}
    for film, film_description_categories in dict_film.items():
        categories = film_description_categories.get("categories")
        for category in categories:
            count_category[category] = count_category.get(category, 0) + 1

        words = film_description_categories.get("description").split()
        if film not in word_description_film:
            word_description_film[film] = {}
        for word in words:
            word_description_film[film][word] = word_description_film[film].get(word, 0) + 1
            if word not in vocab:
                vocab[word] = set()
            vocab[word].add(film)

    for category, count_cate in count_category.items():
        print(category, count_cate)

    total_label = len(dict_film)
    count = {}
    for word in vocab:
        if len(vocab[word]) == total_label:
            count[word] = min([word_description_film[x][word] for x in word_description_film])

    sorted_count = sorted(count, key=count.get, reverse=True)
    for word in sorted_count[:100]:
        print(word, count[word])
    print(len(count_category))
    test_percent = 0.2

    text = []
    label = []

    for film, film_description_categories in dict_film.items():
        label.append(film_description_categories.get("categories"))
        text.append(film_description_categories.get("description"))

    X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=test_percent, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    dict_train = {"description": X_train, "categories": y_train}
    train_data = pandas.DataFrame(data=dict_train)
    train_data.to_csv("train.csv", index=False)

    dict_valid = {"description": X_valid, "categories": y_valid}
    valid_data = pandas.DataFrame(data=dict_train)
    valid_data.to_csv("valid.csv", index=False)

    dict_test = {"description": X_test, "categories": y_test}
    test_data = pandas.DataFrame(data=dict_train)
    test_data.to_csv("test.csv", index=False)
