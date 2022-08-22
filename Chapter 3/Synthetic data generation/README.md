This folder contains the files used to generate synthetic datasets. In this work, four different datasets are used:
* 3D connected volumes:
These volumes are generated as a filled sphere/ellipsoid or a mixture of both shapes.
<a href="url"><img src="https://github.com/RyersonU-DataScienceLab/Sanaz_3D_tumorGAN/blob/main/Synthetic%20data%20generation/Figures/3D_connected_volume.PNG" align="middle" height="300" width="300" ></a>

* 3D connected volumes with packed spheres:
Above volumes have 7 predefined target isocenters. These isocenters are assumed as centers of small spheres and points are filled with different integers to create spheres.

<a href="url"><img src="https://github.com/RyersonU-DataScienceLab/Sanaz_3D_tumorGAN/blob/main/Synthetic%20data%20generation/Figures/connected_7iso.PNG" align="middle" height="200" width="700" ></a>

* 3D connected tumors
3D tumors follow the same distribution of features as the source paper [(Cevik et al, 2018)](https://iopscience.iop.org/article/10.1088/1361-6560/aad105). Those tumors are generated using Matlab and saved as .mat files. main.py loads the .mat files and append them to create a .pkl file as dataset.

<a href="url"><img src="https://github.com/RyersonU-DataScienceLab/Sanaz_3D_tumorGAN/blob/main/Synthetic%20data%20generation/Figures/3D_connected_tumor.PNG" align="middle" height="300" width="300" ></a>

* 3D connected tumors with packed sphere:
Tumors have different sets of isocenters. For each isocenter a small sphere is created.

<a href="url"><img src="https://github.com/RyersonU-DataScienceLab/Sanaz_3D_tumorGAN/blob/main/Synthetic%20data%20generation/Figures/Matlab_8_sio.PNG" align="middle" height="200" width="700" ></a>

Use [main.py](https://github.com/sanazMj/PhD_Thesis_repo/blob/main/Chapter%203/Synthetic%20data%20generation/main.py) to generate a sample dataset. main.py contains two sets of commands for synthetic data generation. 
* 3D connected volume with/without packed spheres generation
  * Define the sample size, dimension and number of isocenters (e.g. 7) you want to create. The created samples will be saved in .h5 file.
```
python main.py --dataset_kind 'iso_sphere_full' --dimension [32,32,32] --dataset_size 100000 --target_points 7 --shape_choice 'Both' --name_choice 'New_generated_data'
```
* 3D tumor volumes with/without packed spheres generation
  * The data samples that follow the same distribution as samples as the source paper are created in Matlab.
  * The .mat files are loaded in python and saved as .h5 files.
```
python main.py --dataset_kind 'Matlab' --dimension [32,32,32] --dataset_size 100000  --name_choice 'New_generated_data_tumor'
```
