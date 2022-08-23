# Detecting_cones_AOSLO
We want to use the idea of the particle systems to detect cones in AOSLO confocal images while following a certain regularity in the way these cones are spread across the image.

# How to start
Install dependencies:
```
pip install -r requirements.txt
```
To detect blobs (AOSLO):
```
python pipeline_with_delitions.py
```
To visualize all real AOSLO position:
```
python visualize_centers.py
```
# Description
### Current algo
1. calculate "blobness" measure using technique offered in [2] /  custom.
2. generate particle in places with ultra hight "blobness"
3. calculate the gradient field of "blobness".
4. fix position of the particles
5. generate particles based on distance energy.
6. move particles in negatieve gradient direction (blobness+distance force).
7. delete particles that comes too close to each other.
8. calculate metrics
9. repeat step 5-8 M-times, but step 5 only every N-run;
 

### ToDo
* Discuss particle-fixing mechnizm
* Optimize


# Papers:
[1] Kindlmann et.al, 2009, "Sampling and Visualizing Creases with Scale-Space Particles", http://people.cs.uchicago.edu/~glk/ssp/

[2] Alejandro et.al, 2000, "Multiscale Vessel Enhancement Filtering", https://www.researchgate.net/publication/2388170_Multiscale_Vessel_Enhancement_Filtering 
 