mkdir -p ./display-grid

# 3D
python observer-cone.py --dimension 3 --display_basis ConeHering --output_filename ./display-grid/3d-cone-basis.mp4 --total_frames 90 --fps 30
python observer-cone.py --dimension 3 --display_basis Hering --output_filename ./display-grid/3d-max-basis.mp4 --total_frames 90 --fps 30


# 4D
python observer-cone.py --dimension 4 --display_basis Cone --output_filename ./display-grid/4d-cone-basis.mp4 --total_frames 90 --fps 30
python observer-cone.py --dimension 4 --display_basis MaxBasis --output_filename ./display-grid/4d-max-basis.mp4 --total_frames 90 --fps 30
