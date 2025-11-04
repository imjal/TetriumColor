mkdir -p ./simplex-grid-outputs
python simplex-elements.py --dimension 3 --template govardovskii --primary_wavelengths 425 515 696 --output_filename ./simplex-grid-outputs/3d-simplex.mp4

# Create 4 viewpoints to label the non-spectral axes
for i in {0..3};
  do python simplex-elements.py --dimension 4 --template govardovskii  --face_idx $i --output_filename ./simplex-grid-outputs/4d-simplex-$i.mp4
done
