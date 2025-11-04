mkdir -p ./metameric-grid

# 4D
python obs.py --dimension 4 --display_basis MaxBasis --output_filename ./metameric-grid/4d-metameric.mp4 --total_frames 90 --fps 30

# 3D
python obs.py --subset 0 1 2 --dimension 3 --display_basis Cone --output_filename ./metameric-grid/3d-metameric-012.mp4 --total_frames 90 --fps 30
python obs.py --subset 0 1 3 --dimension 3 --display_basis Cone --output_filename ./metameric-grid/3d-metameric-013.mp4 --total_frames 90 --fps 30
python obs.py --subset 0 2 3 --dimension 3 --display_basis Cone --output_filename ./metameric-grid/3d-metameric-023.mp4 --total_frames 90 --fps 30
python obs.py --subset 1 2 3 --dimension 3 --display_basis Cone --output_filename ./metameric-grid/3d-metameric-123.mp4 --total_frames 90 --fps 30

# 2D