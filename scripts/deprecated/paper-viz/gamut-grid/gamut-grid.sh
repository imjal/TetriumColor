mkdir -p ./outputs

# 3D 
python gamut.py --dimension 3 --display_basis Hering --primary_wavelengths 425 515 695 --ideal --output_file ./outputs/3d_Hering_ideal.mp4 --fps 30 --total_frames 90
python gamut.py --dimension 3 --display_basis Hering --output_file ./outputs/3d_Hering_Tutenlab.mp4 --fps 30 --total_frames 90

python gamut.py --dimension 3 --display_basis Hering --primary_wavelengths 450 535 600 --ideal --output_file ./outputs/3d_Hering_viz_eff.mp4 --fps 30 --total_frames 90


# 4D
python gamut.py --dimension 4 --step_size 10 --display_basis MaxBasis --primary_wavelengths 410 510 585 695 --ideal --output_file ./outputs/4d_Hering_ideal.mp4 --fps 30 --total_frames 90
python gamut.py --dimension 4 --step_size 10 --display_basis MaxBasis --primary_wavelengths 445 530 585 630  --ideal --output_file ./outputs/4d_Hering_viz_eff.mp4 --fps 30 --total_frames 90
python gamut.py --dimension 4 --step_size 10 --display_basis MaxBasis --output_file ./outputs/4d_Hering_Tutenlab.mp4 --fps 30 --total_frames 90