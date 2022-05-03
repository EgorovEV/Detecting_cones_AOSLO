# Detecting_cones_AOSLO
We want to use the idea of the particle systems to detect cones in AOSLO confocal images while following a certain regularity in the way these cones are spread across the image.

# How to start
Install dependencies:
```
code will be here
```
To detect blobs (AOSLO):
```
python main.py
```
To visualize real AOSLO position:
```
python visualize_centers.py
```
# Description
### Current algo
1. create N particles.
2. calculate "blobness" measure using technique offered in [2].
3. calculate the gradient field of "blobness".
4. move particles in negatieve gradient direction.
5. visualize current particle position.
6. repeat step 4 M-times, every 100 updates perform step 5. 
### Also implemented
* finding and calculating distance to N nearest particles.
* writing visualization of algo results into ./examples dir.

### ToDo
* Include distance to the algo
* Discuss posible metric to measure algo perfomance


# Papers:
[1] Kindlmann et.al, 2009, "Sampling and Visualizing Creases with Scale-Space Particles", http://people.cs.uchicago.edu/~glk/ssp/

[2] Alejandro et.al, 2000, "Multiscale Vessel Enhancement Filtering", https://www.researchgate.net/publication/2388170_Multiscale_Vessel_Enhancement_Filtering 
 