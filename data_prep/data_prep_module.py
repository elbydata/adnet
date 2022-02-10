import pandas as pd
import numpy as np
from unidecode import unidecode
import string
import os
import gc
from tqdm import tqdm


def alt_countries():

    alt_countries = {
        "austria": ["austria", "osterreich", "at", "aut"],
        "belgium": ["belgium", "belgie", "belgique", "be", "bel"],
        "czech_republic": [
            "czech_republic",
            "cesko",
            "czechia",
            "cz",
            "cze",
        ],
        "denmark": ["denmark", "danmark", "dk", "dnk"],
        "estonia": ["estonia", "eesti", "ee", "est"],
        "finland": ["finland", "suomi", "fi", "fin"],
        "france": ["france", "fr", "fra"],
        "germany": ["germany", "deutschland", "de", "deu"],
        "greece": ["greece", "ellada", "ellas", "gr", "grc"],
        "italy": ["italy", "italia", "it", "ita"],
        "latvia": ["latvia", "latvija", "lv", "lva"],
        "lithuania": ["lithuania", "lietuva", "lt", "ltu"],
        "luxembourg": [
            "luxembourg",
            "letzebuerg",
            "luxemburg",
            "lu",
            "lux",
        ],
        "netherlands": [
            "netherlands",
            "nederland",
            "holland",
            "nl",
            "nld",
        ],
        "poland": ["poland", "polska", "pl", "pol"],
        "portugal": ["portugal", "pt", "prt"],
        "romania": ["romania", "ro", "rou"],
        "slovakia": ["slovakia", "slovensko", "sk", "svk"],
        "slovenia": ["slovenia", "slovenija", "si", "svn"],
        "spain": ["spain", "espana", "es", "esp"],
        "sweden": ["sweden", "sverige", "se", "swe"],
    }

    return alt_countries


def data_preprocess(country, base_path, max_rows):

    rng = np.random.default_rng(42)

    # load data
    df = pd.read_csv(f"{base_path}{country}.csv", dtype=str)
    gc.collect()

    # drop rows with NaN values for number, street, or postcode columns
    df = df.dropna(subset=["NUMBER", "STREET", "POSTCODE"])

    # drop duplicate rows
    df.drop_duplicates(inplace=True)

    # drop rows containing overly long items (to use as necessary)
    target_col = "STREET"
    target_max = 50
    df.drop(
        df.loc[df[target_col].map(len) > target_max].index, inplace=True
    )

    # shuffle row order
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # take subset if greater than max length
    df_len = df.shape[0]
    if df_len > max_rows:
        df = df[0:max_rows]
    else:
        pass

    return df


def batch_df(df, batch_size):
    batches = []
    n_batches = len(df) // batch_size + (
        1 if len(df) % batch_size else 0
    )
    for i in range(n_batches):
        batches.append(df[i * batch_size : (i + 1) * batch_size])
    return batches


def batch_process(batches, country, seq_size):

    rng = np.random.default_rng(42)

    X_arrays = []
    y_arrays = []

    for df in tqdm(batches, total=len(batches)):

        # drop unused columns, fill NaN values, convert to string type
        df.drop(
            ["LON", "LAT", "UNIT", "DISTRICT", "REGION", "ID", "HASH"],
            axis=1,
            inplace=True,
        )
        df.fillna("_", inplace=True)
        df = df.astype(str)

        # import alternative country names
        alt_names = alt_countries()[country]

        # random choice of alternative country name
        name_choice = [
            alt_names[i]
            for i in rng.choice(len(alt_names), df.shape[0])
        ]

        # add "country" column
        df["COUNTRY"] = name_choice

        # convert special characters standard characters and convert to lowercase
        df = df.applymap(lambda x: unidecode(x))
        df = df.applymap(lambda x: x.lower())

        # pad values to max length for each column
        num_max_len = max([len(i) for i in list(set(df["NUMBER"]))])
        street_max_len = max([len(i) for i in list(set(df["STREET"]))])
        city_max_len = max([len(i) for i in list(set(df["CITY"]))])
        postcode_max_len = max(
            [len(i) for i in list(set(df["POSTCODE"]))]
        )
        country_max_len = 14
        df["NUMBER"] = df["NUMBER"].str.pad(
            width=num_max_len, side="both", fillchar="_"
        )
        df["STREET"] = df["STREET"].str.pad(
            width=street_max_len, side="both", fillchar="_"
        )
        df["CITY"] = df["CITY"].str.pad(
            width=city_max_len, side="both", fillchar="_"
        )
        df["POSTCODE"] = df["POSTCODE"].str.pad(
            width=postcode_max_len, side="both", fillchar="_"
        )
        df["COUNTRY"] = df["COUNTRY"].str.pad(
            width=country_max_len, side="both", fillchar="_"
        )

        # create labels df and fill values with class labels
        df_labels = df.copy()
        df_labels["NUMBER"] = df_labels["NUMBER"].apply(
            lambda x: "".join([i if i == "_" else "1" for i in list(x)])
        )
        df_labels["STREET"] = df_labels["STREET"].apply(
            lambda x: "".join([i if i == "_" else "2" for i in list(x)])
        )
        df_labels["CITY"] = df_labels["CITY"].apply(
            lambda x: "".join([i if i == "_" else "3" for i in list(x)])
        )
        df_labels["POSTCODE"] = df_labels["POSTCODE"].apply(
            lambda x: "".join([i if i == "_" else "4" for i in list(x)])
        )
        df_labels["COUNTRY"] = df_labels["COUNTRY"].apply(
            lambda x: "".join([i if i == "_" else "5" for i in list(x)])
        )

        # create data and label arrays
        X = df.to_numpy()
        y = df_labels.to_numpy()

        # free memory
        del df
        del df_labels
        gc.collect()

        # create shuffle index for columns
        shuf = np.stack(
            [
                rng.choice(X.shape[1], X.shape[1], replace=False)
                for i in range(X.shape[0])
            ]
        )

        # shuffle columns
        X = np.take_along_axis(X, shuf, 1)
        y = np.take_along_axis(y, shuf, 1)

        # free memory
        del shuf
        gc.collect()

        # expand array contents
        X = np.array([[i for s in k for i in s] for k in X])
        y = np.array([[i for s in k for i in s] for k in y])

        # fill blank labels with zero
        y[y == "_"] = 0

        # import ascii characters for random padding
        chars = (
            string.ascii_lowercase + string.digits + string.punctuation
        )
        char_array = np.array(
            [i for i in list(chars) if i not in ("\\")]
        )

        # pad sequences and fill blanks with random characters
        X_left_pad = divmod(seq_size - X.shape[1], 2)[0]
        X_right_pad = X_left_pad + divmod(seq_size - X.shape[1], 2)[1]
        X = np.pad(
            X,
            [(0, 0), (X_left_pad, X_right_pad)],
            mode="constant",
            constant_values="_",
        )
        size_to_fill = (X == "_").sum()
        random_chars = rng.choice(char_array, size_to_fill)
        X[X == "_"] = random_chars

        # pad labels and with zeros
        y_left_pad = divmod(seq_size - y.shape[1], 2)[0]
        y_right_pad = X_left_pad + divmod(seq_size - y.shape[1], 2)[1]
        y = np.pad(
            y,
            [(0, 0), (y_left_pad, y_right_pad)],
            mode="constant",
            constant_values=0,
        )
        y = y.astype(np.int)

        X_arrays.append(X)
        y_arrays.append(y)

    X = np.concatenate(X_arrays, axis=0)
    y = np.concatenate(y_arrays, axis=0)

    return X, y


def save_arrays(X, y, target_path, country):

    for i in ["X", "y"]:
        if os.path.exists(target_path):
            if os.path.exists(f"{target_path}{i}"):
                pass
            else:
                os.mkdir(f"{target_path}{i}")
        else:
            os.mkdir(target_path)
            os.mkdir(f"{target_path}{i}")

    # compress and save arrays
    np.savez_compressed(f"{target_path}X/X_{country}", X)
    np.savez_compressed(f"{target_path}y/y_{country}", y)


def display_random(X, y):

    rng = np.random.default_rng(42)

    # display random X and y value and sense check values
    display_idx = rng.choice(X.shape[0])
    for k, v in {"X": X, "y": y}.items():
        print(f"{k}_sample:")
        print(v[display_idx])
        print("")
    for i in range(1, 6):
        print("".join(X[display_idx][np.where(y[display_idx] == i)[0]]))


def adnet_processing(
    country,
    base_path,
    target_path,
    max_rows,
    seq_size,
    batch_size,
    display=True,
    save=True,
):

    print(f"Processing {country} data...")
    df = data_preprocess(country, base_path, max_rows)
    print("Batch processing...")
    batches = batch_df(df, batch_size)
    X, y = batch_process(batches, country, seq_size)
    if display == True:
        display_random(X, y)
    if save == True:
        print("")
        print("Saving arrays...")
        save_arrays(X, y, target_path, country)
        print(f"Arrays saved in {target_path}.")
