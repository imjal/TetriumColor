from typing import List, Dict
import numpy.typing as npt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from TetriumColor.Observer import Spectra, Illuminant, Observer
from TetriumColor.Observer.Inks import InkLibrary, InkGamut


def analyze_inkset(all_inks: Dict[str, Spectra], paper: Spectra, wavelengths: npt.NDArray) -> List[npt.NDArray]:

    # Initialize the ink library
    full_library = InkLibrary(all_inks, paper)

    # Define observer and illuminant
    d65 = Illuminant.get("d65")
    tetrachromat = Observer.tetrachromat(illuminant=d65, wavelengths=wavelengths)

    # Perform convex hull search
    top_volumes_all_inks = full_library.convex_hull_search(tetrachromat, d65)

    chosen_idx = 1
    best4 = [all_inks[ink_name] for ink_name in top_volumes_all_inks[chosen_idx][1]]
    gamut = InkGamut(best4, paper, d65)

    spectra = gamut.get_spectra([0, 0, 1, 1])

    # top_volumes_all_inks = full_library.cached_pca_search(tetrachromat, d65, k=4)

    # Display top results
    print("Top ink combinations by volume:")
    for volume, inks in top_volumes_all_inks[:10]:
        print(f"Volume: {volume}, Inks: {inks}")

    # Plot reflectance spectra of top inks
    plt.figure(figsize=(10, 6))
    for ink_name in top_volumes_all_inks[0][1]:  # Top ink combination
        all_inks[ink_name].plot()
    plt.legend()
    plt.title("Reflectance Spectra of Top Inks")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.show()

    return top_volumes_all_inks


def analyze_ansari():
    ### Analyze the Ansari inkset ###
    # Load the CSV data
    data_path = "../../data/ansari/combined_ink_library.csv"
    df = pd.read_csv(data_path)

    spectras = df.iloc[:, 5:].to_numpy()  # Extract reflectance data
    wavelengths = np.arange(400, 701, 10)  # Wavelengths from 400 to 700 nm in steps of 10 nm

    # Create Spectra objects for each ink
    all_inks = {}
    for i in range(spectras.shape[0]):
        name = "".join([str(df.iloc[i, j]) for j in range(3)])
        all_inks[name] = Spectra(data=spectras[i], wavelengths=wavelengths)

    paper = all_inks.popitem()[1]  # last ink is the paper?

    top_volumes_all_inks = analyze_inkset(all_inks, paper, wavelengths)
    # Save results to CSV
    results_df = pd.DataFrame(top_volumes_all_inks, columns=['Volume', 'Inks'])
    results_df.to_csv('top_ink_combinations_ansari.csv', index=False)

    print("Analysis on Ansari is complete. Top ink combinations by volume have been printed.")


def analyze_fp_inks():
    ### Analyze our 100 ink gamut ###
    # Load the CSV data
    data_path = "../../data/fp_inks/all_inks.csv"
    df = pd.read_csv(data_path)

    spectras = df.iloc[:, 1:].to_numpy()  # Extract reflectance data
    wavelengths = np.arange(400, 701, 10)  # Wavelengths from 400 to 700 nm in steps of 10 nm
    # Create Spectra objects for each ink
    all_inks = {}
    for i in range(spectras.shape[0]):
        name = "".join([str(df.iloc[i, j]) for j in range(3)])
        all_inks[name] = Spectra(data=spectras[i], wavelengths=wavelengths)

    paper = all_inks.popitem()[1]  # las
    top_volumes_all_inks = analyze_inkset(all_inks, paper, wavelengths)
    # Save results to CSV
    results_df = pd.DataFrame(top_volumes_all_inks, columns=['Volume', 'Inks'])
    results_df.to_csv('top_ink_combinations_ours.csv', index=False)

    print("Analysis on our 100 ink gamut is complete. Top ink combinations by volume have been printed.")


def main():

    analyze_ansari()
    # analyze_fp_inks()

    print("All analyses completed successfully.")


if __name__ == "__main__":
    main()
