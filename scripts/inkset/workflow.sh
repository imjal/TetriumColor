# How to print metameric ishihara plates

# Goal: make this exist only in the printer code (don't use two repositories), 
# and have one script that will go through this entire process without 
# having to exit. 

# 1. measure all of the ink combinations, and save to csv.
# call print_calibration_target function in measure_library
# get a csv file of all of the ink combination -> spectra ( ex: "C255 M255 O255" -> Spectra) 
python measure_library.py # in ESCP Printer Code


# 2.1 Register Ink Library (just CMYO)
python ink_processing.py create --format nix lib_name files.csv # in TetriumColor scripts/inkset/

# 2.2 Calibrate ink model (transfer function), output a calibration json. 
# call ink_calibrate.py calibrate with the data of single ink spectras & the above file 
python ink_calibrate.py calibrate CMYO-10-6 data/measurements/2025-10-12/CMYO-10-12.csv # in TetriumColor scripts/inkset/

# 3. Make ink gamut with the transfer function and find metamers (find-metamers.ipynb) # in TetriumColor scripts/inkset/
ink_gamut = InkGamut(primaries_dict, primaries_dict["0000"], calibration_json="./results/model-10-12.json") 
observer = Observer.tetrachromat(illuminant=illuminant)


# 4. Print Ishihara plates and loop the above. 
# call the print_ishihara_test(percentages) function
python measure_library.py # in ESCP Printer Code 