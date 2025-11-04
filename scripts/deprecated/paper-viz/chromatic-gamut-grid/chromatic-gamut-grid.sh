mkdir -p ./chromatic-gamut-grid-outputs

# 3D 
python chromatic-gamut.py --dimension 3 --display_type ideal --max_simplex_wavelengths 425 515 695  --output_file ./chromatic-gamut-grid-outputs/3d-ideal-simplex.mp4 --fps 30 --total_frames 90
python chromatic-gamut.py --dimension 3 --display_type max-perceptual --max_simplex_wavelengths 425 515 695 --max_perceptual_volume_wavelengths 440 545 620  --output_file ./chromatic-gamut-grid-outputs/3d-max-perceptual.mp4 --fps 30 --total_frames 90
python chromatic-gamut.py --dimension 3 --display_type ours --max_simplex_wavelengths 425 515 695  --output_file ./chromatic-gamut-grid-outputs/3d-our-gamut.mp4 --fps 30 --total_frames 90

# 4D
python chromatic-gamut.py --dimension 4 --template govardovskii --display_type ideal --max_simplex_wavelengths 410 510 585 695 --output_file ./chromatic-gamut-grid-outputs/4d-ideal-simplex.mp4 --fps 30 --total_frames 90
python chromatic-gamut.py --dimension 4 --template govardovskii --display_type max-perceptual  --max_simplex_wavelengths 410 510 585 695 --max_perceptual_volume_wavelengths 435 515 610 660  --output_file ./chromatic-gamut-grid-outputs/4d-max-perceptual.mp4 --fps 30 --total_frames 90
python chromatic-gamut.py --dimension 4 --template govardovskii --display_type ours  --max_simplex_wavelengths 410 510 585 695  --output_file ./chromatic-gamut-grid-outputs/4d-our-gamut.mp4 --fps 30 --total_frames 90