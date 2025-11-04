mkdir -p ./outputs
# 3D
python anomalous-mismatch.py --display_basis MaxBasis --mismatch -1 --output_filename ./outputs/anomalous-mismatch-all.mp4 --total_frames 90 --fps 30
python anomalous-mismatch.py --display_basis Hering  --step_size 1 --mismatch 0 --output_filename ./outputs/anomalous-mismatch-0.mp4 --total_frames 90 --fps 30
python anomalous-mismatch.py --display_basis Hering  --step_size 1 --mismatch 1 --output_filename ./outputs/anomalous-mismatch-1.mp4 --total_frames 90 --fps 30
python anomalous-mismatch.py --display_basis Hering  --step_size 1 --mismatch 2 --output_filename ./outputs/anomalous-mismatch-2.mp4 --total_frames 90 --fps 30