import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None


def get_location_names():
    return __locations


def get_data_columns():
    return __data_columns


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[2:]  # first 3 columns are sqft, bhk

    global __model
    if __model is None:
        with open('./artifacts/mumbai_price_prediction_model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")


def get_estimated_price(location, area, bhk):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = area
    x[1] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0] / 100000, 2)


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('Airoli', 1000, 3))
    print(get_estimated_price('Airoli', 1000, 2))
    print(get_estimated_price('Andheri West', 1000, 2))  # other location
    print(get_estimated_price('Andheri East', 1000, 2))  # other location
