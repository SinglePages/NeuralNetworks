import numpy as np
import pandas as pd
from pathlib import Path
from regex import search
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


@st.cache
def load_data():
    data_dir = Path("../Data/weather/")

    data_header_filename = data_dir / "HEADERS.txt"
    data2020_dirname = data_dir / "2020"

    with open(data_header_filename) as data_header_file:
        data_header = [line.split() for line in data_header_file]

    all_header_names = data_header[1]
    for i, name in enumerate(all_header_names):
        print(f"{i:>2} : {name}")

    cols_to_keep = [0, 1, 2, 3, 4, 6, 7, 9, 13, 20, 26]
    # solar radiation? 13, 14, 15, 16, 17, 18
    # infrared surface temperature 19, 20, 21, 22, 23, 24, 25
    # relative humidity 26, 27
    # soild moisture 28, 29, 30, 31, 32
    # soild temperature 33, 34, 35, 36, 37

    header_names = [
        h.lower() for i, h in enumerate(all_header_names) if i in cols_to_keep
    ]

    # Missing data given as -9999.0 for 7-character fields with one decimal
    # and -99.000 for 7-charcter fields with three decimal places

    na_values = ["-9999.0", "-99.000"]

    count = 0

    state_dfs = []
    for data_filename in data2020_dirname.glob("*.txt"):
        print("Reading", data_filename.name, end=" ")
        if match := search(
            "^CRNH.*-(\d{4})-([A-Z]+)_([A-Za-z]+).*", data_filename.name
        ):
            year, state, site = match.group(1), match.group(2), match.group(3)
            print(":", year, state, site)
        else:
            print("No match")
            break

        if state == "AK" or state == "HI":
            print("***Skipping", state)
            continue

        df = pd.read_csv(
            data_filename,
            names=header_names,
            usecols=cols_to_keep,
            delim_whitespace=True,
            na_values=na_values,
        )

        df["state"] = state
        df["site"] = site

        state_dfs.append(df)
        count += 1

    all_states_df = pd.concat(state_dfs, ignore_index=True)
    all_states_df = all_states_df.dropna()

    return all_states_df


@st.cache
def split_data(data, feature_cols, output_cols):
    X = data.loc[:, feature_cols]
    y = data.loc[:, output_cols]
    return train_test_split(X, y, random_state=0)


@st.cache
def train_model(X, y):

    pipe = make_pipeline(StandardScaler(), Ridge())

    pipe.fit(X, y)

    return pipe


@st.cache
def unique_stations(data):
    return data.groupby(["latitude", "longitude"]).size().reset_index()


data_load_state = st.text("Loading data...")
data = load_data()
data_load_state.text("Loading data...done!")


feature_cols = [
    "utc_date",
    "utc_time",
    "lst_date",
    "lst_time",
    "t_hr_avg",
    "solarad",
    "sur_temp",
    "rh_hr_avg",
]

output_cols = ["latitude", "longitude"]

X_train, X_valid, y_train, y_valid = split_data(data, feature_cols, output_cols)
lat_model = train_model(X_train, y_train["latitude"])
lon_model = train_model(X_train, y_train["longitude"])

station_geos = unique_stations(data)

if st.checkbox("Show head of dataframe"):
    st.write(data.head())

if st.checkbox("Show summary of dataframe"):
    st.write(data.describe())

if st.checkbox("Show Example"):
    f"""
    Latitude: [{data["latitude"].min()}, {data["latitude"].max()}]
    Longitude: [{data["longitude"].min()}, {data["longitude"].max()}]
    """

    valid_idx = np.random.randint(X_valid.shape[0])
    if st.button("Random validation example"):
        valid_idx = np.random.randint(X_valid.shape[0])

    ylat_true, ylon_true = y_valid.iloc[valid_idx]
    ylat_pred = lat_model.predict(X_valid.iloc[valid_idx].to_numpy().reshape(1, -1))[0]
    ylon_pred = lon_model.predict(X_valid.iloc[valid_idx].to_numpy().reshape(1, -1))[0]

    st.write(f"Latitude prediction: {ylat_pred} (True value: {ylat_true})")
    st.write(f"Longitude prediction: {ylon_pred} (True value: {ylon_true})")

    guess_data = pd.DataFrame(
        {"lat": [ylat_true, ylat_pred], "lon": [ylon_true, ylon_pred]}
    )

    st.map(guess_data, zoom=3)

    if st.checkbox("Show map"):
        st.map(station_geos)
