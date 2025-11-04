mkdir -p ./display-grid
# 2D
python observer-cone-fundamentals.py --dimension 2 --template govardovskii --output_filename ./display-grid/2d-cone-fundamentals.csv
python observer-cone-chromaticity.py --dimension 2 --chrom_type CHROM --template govardovskii --output_filename ./display-grid/2d-chromaticity-cone.mp4
python observer-cone-chromaticity.py --dimension 2 --chrom_type HERING_CHROM --template govardovskii --output_filename ./display-grid/2d-chromaticity-max-basis.mp4
python observer-perceptual.py --dimension 2 --polyscope_display_type HERING_CONE_PERCEPTUAL_300 --template govardovskii --output_filename ./display-grid/2d-perceptual-cone.mp4
python observer-perceptual.py --dimension 2 --polyscope_display_type HERING_MAXBASIS_PERCEPTUAL_300 --template govardovskii --output_filename ./display-grid/2d-perceptual-maxbasis.mp4
# python observer-max-simplex.py --dimension 2 --template govardovskii --primary_wavelengths 410 615 --output_filename ./display-grid/2d-max-simplex.mp4 --total_frames 90 --fps 30

# 3D
python observer-cone-fundamentals.py --dimension 3 --template govardovskii --output_filename ./display-grid/3d-cone-fundamentals.csv
python observer-cone-chromaticity.py --dimension 3 --chrom_type CHROM --template govardovskii --output_filename ./display-grid/3d-chromaticity-cone.mp4
python observer-cone-chromaticity.py --dimension 3 --chrom_type HERING_CHROM  --template govardovskii --output_filename ./display-grid/3d-chromaticity-max-basis.mp4
python observer-perceptual.py --dimension 3 --polyscope_display_type HERING_CONE_PERCEPTUAL_300 --template govardovskii --output_filename ./display-grid/3d-perceptual-cone.mp4
python observer-perceptual.py --dimension 3 --polyscope_display_type HERING_MAXBASIS_PERCEPTUAL_300 --template govardovskii --output_filename ./display-grid/3d-perceptual-maxbasis.mp4
# python observer-max-simplex.py --dimension 3 --template govardovskii --primary_wavelengths 425 515 695 --output_filename ./display-grid/3d-max-simplex.mp4 --total_frames 90 --fps 30

# 4D
python observer-cone-fundamentals.py --dimension 4 --template govardovskii --output_filename ./display-grid/4d-cone-fundamentals.csv
python observer-cone-chromaticity.py --dimension 4 --chrom_type CHROM --template govardovskii --output_filename ./display-grid/4d-chromaticity-cone.mp4 --total_frames 90 --fps 30
python observer-cone-chromaticity.py --dimension 4 --chrom_type HERING_CHROM --template govardovskii --output_filename ./display-grid/4d-chromaticity-max-basis.mp4 --total_frames 90 --fps 30
python observer-perceptual.py --dimension 4 --polyscope_display_type HERING_CONE_PERCEPTUAL_300 --template govardovskii --output_filename ./display-grid/4d-perceptual-cone.mp4
python observer-perceptual.py --dimension 4 --polyscope_display_type HERING_MAXBASIS300_PERCEPTUAL_300 --template govardovskii --output_filename ./display-grid/4d-perceptual-maxbasis.mp4
# python observer-max-simplex.py --dimension 4 --template govardovskii --primary_wavelengths 410 510 585 695 --output_filename ./display-grid/4d-max-simplex.mp4 --total_frames 90 --fps 30