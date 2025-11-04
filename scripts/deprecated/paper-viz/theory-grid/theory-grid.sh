mkdir -p ./display-grid
# 2D
python observer-cone-fundamentals.py --dimension 2 --template govardovskii --output_filename ./display-grid/2d-cone-fundamentals.csv
python observer-cone.py --dimension 2 --display_basis ConeHering --output_filename ./display-grid/2d-cone-basis.mp4 --total_frames 90 --fps 30
python observer-cone.py --dimension 2 --display_basis Hering --output_filename ./display-grid/2d-max-basis.mp4 --total_frames 90 --fps 30
python observer-cone-chromaticity.py --dimension 2 --display_basis Hering --template govardovskii --output_filename ./display-grid/2d-chromaticity-max-basis.mp4
python observer-max-simplex.py --dimension 2 --template govardovskii --primary_wavelengths 410 615 --output_filename ./display-grid/2d-max-simplex.mp4 --total_frames 90 --fps 30

# 3D
python observer-cone-fundamentals.py --dimension 3 --template govardovskii --output_filename ./display-grid/3d-cone-fundamentals.csv
python observer-cone.py --dimension 3 --display_basis ConeHering --output_filename ./display-grid/3d-cone-basis.mp4 --total_frames 90 --fps 30
python observer-cone.py --dimension 3 --display_basis Hering --output_filename ./display-grid/3d-max-basis.mp4 --total_frames 90 --fps 30
python observer-cone-chromaticity.py --dimension 3 --display_basis Hering --template govardovskii --output_filename ./display-grid/3d-chromaticity-max-basis.mp4
python observer-max-simplex.py --dimension 3 --template govardovskii --primary_wavelengths 425 515 695 --output_filename ./display-grid/3d-max-simplex.mp4 --total_frames 90 --fps 30

# 4D
python observer-cone-fundamentals.py --dimension 4 --template govardovskii --output_filename ./display-grid/4d-cone-fundamentals.csv
python observer-cone.py --dimension 4 --display_basis Cone --output_filename ./display-grid/4d-cone-basis.mp4 --total_frames 90 --fps 30
python observer-cone.py --dimension 4 --display_basis MaxBasis --output_filename ./display-grid/4d-max-basis.mp4 --total_frames 90 --fps 30
python observer-cone-chromaticity.py --dimension 4 --display_basis Hering --template govardovskii --output_filename ./display-grid/4d-chromaticity-max-basis.mp4 --total_frames 90 --fps 30
python observer-max-simplex.py --dimension 4 --template govardovskii --primary_wavelengths 410 510 585 695 --output_filename ./display-grid/4d-max-simplex.mp4 --total_frames 90 --fps 30