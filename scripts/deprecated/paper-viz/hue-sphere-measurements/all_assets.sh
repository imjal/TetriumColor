mkdir -p ./results/

python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 1.0 --output_filename ./results/ball_q.mp4
python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 0.5 --lattice  --output_filename ./results/ball_lattice_q.mp4
python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --lattice --output_filename ./results/lattice_q.mp4

python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 1.0 --metameric_dir --output_filename ./results/ball_dir_q.mp4
python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 0.5 --lattice  --metameric_dir --output_filename ./results/ball_lattice_dir_q.mp4
python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --lattice --metameric_dir --output_filename ./results/lattice_dir_q.mp4

# python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 1.0 --which_dir saq  --output_filename ./results/ball_saq.mp4
# python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 0.5 --lattice  --which_dir saq --output_filename ./results/ball_lattice_saq.mp4
# python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --lattice --which_dir saq --output_filename ./results/lattice_saq.mp4

# python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 1.0 --which_dir saq --metameric_dir --output_filename ./results/ball_dir_saq.mp4
# python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 0.5 --lattice --which_dir saq --metameric_dir --output_filename ./results/ball_lattice_dir_saq.mp4
# python renderHueSphereAsset.py --display_basis HERING_MAXBASIS  --lattice  --which_dir saq --metameric_dir --output_filename ./results/lattice_dir_saq.mp4

# with shadows
python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 1.0 --shadow --output_filename ./results/ball_q_shadow.mp4
python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 0.5 --lattice --shadow --output_filename ./results/ball_lattice_q_shadow.mp4
python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --lattice --shadow --output_filename ./results/lattice_q_shadow.mp4

python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 1.0 --metameric_dir --shadow --output_filename ./results/ball_dir_q_shadow.mp4
python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 0.5 --lattice --metameric_dir --shadow --output_filename ./results/ball_lattice_dir_q_shadow.mp4
python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --lattice --metameric_dir --shadow --output_filename ./results/lattice_dir_q_shadow.mp4

# python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 1.0 --which_dir saq --shadow --output_filename ./results/ball_saq_shadow.mp4
# python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 0.5 --lattice --which_dir saq --shadow --output_filename ./results/ball_lattice_saq_shadow.mp4
# python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --lattice --which_dir saq --shadow --output_filename ./results/lattice_saq_shadow.mp4

# python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 1.0 --which_dir saq --metameric_dir --shadow --output_filename ./results/ball_dir_saq_shadow.mp4
# python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --ball --transparency_ball 0.5 --lattice --which_dir saq --metameric_dir --shadow --output_filename ./results/ball_lattice_dir_saq_shadow.mp4
# python renderHueSphereAsset.py --display_basis HERING_MAXBASIS --lattice --which_dir saq --metameric_dir --shadow --output_filename ./results/lattice_dir_saq_shadow.mp4