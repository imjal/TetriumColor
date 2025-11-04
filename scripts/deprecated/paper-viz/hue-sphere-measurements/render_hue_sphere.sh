mkdir -p ./results/

python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 1.0 --which_dir none --rotation_axis 0 1 0  --output_filename ./results/ball_new.mp4
python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 0.5  --lattice --which_dir none --rotation_axis 0 1 0  --output_filename ./results/ball_lattice_new.mp4
python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --lattice  --which_dir none --rotation_axis 0 1 0 --output_filename ./results/lattice_new.mp4

# with shadows
python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 1.0 --which_dir none --rotation_axis 0 1 0  --shadow --output_filename ./results/ball_new_shadow.mp4
python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 0.5  --lattice --which_dir none --rotation_axis 0 1 0  --shadow  --output_filename ./results/ball_lattice_new_shadow.mp4
python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --lattice  --which_dir none --rotation_axis 0 1 0  --shadow  --output_filename ./results/lattice_new_shadow.mp4
