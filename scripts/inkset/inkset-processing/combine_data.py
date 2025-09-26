import numpy as np
import pandas as pd
from scipy.io import loadmat

import csv
from TetriumColor.Observer import Spectra


def combine_ansari_data():

    # Load the reflectance library from the .mat file
    data = loadmat("../../data/inksets/ansari/complete_reflectance_library.mat")
    reflectance_data = data["reflectance"]
    wavelengths = np.arange(400, 701, 10)

    # Load the ink library description from the .csv file
    description_df = pd.read_csv("../../data/inksets/ansari/ink_library_description.csv")

    # Combine the data into a single DataFrame
    reflectance_df = pd.DataFrame(reflectance_data.T, index=wavelengths)
    reflectance_df.index.name = "Wavelength (nm)"

    # Merge the reflectance data with the description data
    combined_df = description_df.merge(reflectance_df.T, left_index=True, right_index=True)

    # Combine the metadata columns into a single column
    combined_df['Name'] = (
        combined_df['Color'].astype(str) + ', ' +
        combined_df['Manufacturer'].astype(str) + ', ' +
        combined_df['Type'].astype(str) + ', ' +
        combined_df['Details'].astype(str)
    )

    # Reorder columns to put Name as the second column
    cols = combined_df.columns.tolist()
    cols.remove('Name')
    cols.insert(1, 'Name')
    combined_df = combined_df[cols]

    # Drop the original metadata columns
    combined_df = combined_df.drop(['Color', 'Manufacturer', 'Type', 'Details'], axis=1)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv("../../data/inksets/ansari/ansari-inks.csv", index=False)

    # Display the first few rows of the combined DataFrame
    print(combined_df.head())


def combine_fp_inks_data():
    wavelengths10 = np.arange(400, 701, 10)  # Wavelengths from 400 to 700 nm in steps of 10 nm
    all_inks = {}
    with open(f'../../data/nix/02022024.csv') as csvf:
        spamreader = csv.reader(csvf, delimiter=';')
        for i, row in enumerate(spamreader):
            if i < 4:
                continue
            name = row[4]
            color_data = np.array(row[33:], dtype=float)
            spectra = Spectra(data=color_data, wavelengths=wavelengths10)
            all_inks[name] = spectra

    with open(f'../../data/nix/011624.csv') as csvf:
        spamreader = csv.reader(csvf, delimiter=';')
        for i, row in enumerate(spamreader):
            if i < 4:
                continue
            name = row[4]
            color_data = np.array(row[33:], dtype=float)
            try:
                spectra = Spectra(data=color_data, wavelengths=wavelengths10)
            except ValueError:
                continue
            all_inks[name] = spectra

    with open(f'../../data/nix/Inks_all.csv') as csvf:
        spamreader = csv.reader(csvf, delimiter=';')
        for i, row in enumerate(spamreader):
            if i < 4:
                continue
            name = row[4]
            color_data = np.array(row[33:], dtype=float)
            spectra = Spectra(data=color_data, wavelengths=wavelengths10)
            all_inks[name] = spectra

    cmy_primaries_dict = {}
    primary_fns = [
        "000",
        "001",
        "010",
        "100",
        "011",
        "110",
        "101",
        "111",
    ]

    for fn in primary_fns:
        with open(f'../../data/nix/PrintColors/{fn}.csv') as csvf:
            spamreader = csv.reader(csvf, delimiter=';')
            for i, row in enumerate(spamreader):
                if i == 4:
                    color_data = np.array(row[33:], dtype=float)
                    spectra = Spectra(data=color_data, wavelengths=wavelengths10)
                    cmy_primaries_dict[fn] = spectra

    all_inks["epson cyan"] = cmy_primaries_dict["100"]
    all_inks["epson magenta"] = cmy_primaries_dict["010"]
    all_inks["epson yellow"] = cmy_primaries_dict["001"]
    all_inks["paper"] = cmy_primaries_dict["000"]

    # del all_inks["Noodlers Firefly"]
    # del all_inks["PR Neon Yellow"]

    # Convert all_inks dictionary to DataFrame and save as CSV
    ink_names = list(all_inks.keys())
    wavelengths = np.arange(400, 701, 10)
    reflectance_data = np.array([all_inks[name].data for name in ink_names])

    # Create DataFrame with ink names as columns and wavelengths as rows
    df = pd.DataFrame(reflectance_data, index=ink_names, columns=wavelengths)
    df.index.name = "Wavelength"

    # Save to CSV
    df.to_csv("../../data/fp_inks/all_inks.csv")
    print("Combined FP inks data saved to ../../data/inksets/fp_inks/all_inks.csv")


if __name__ == "__main__":
    combine_ansari_data()
    # combine_fp_inks_data()
