# NDVI PIPELINE SCRIPT

## Script to Download Sentinel Imagery, calculate NDVI and plot the change in time.

### Description

Creating a python script that utilizes sentinelhub API to download sentinel satellite images for a 
specified area of interest (AOI) in geojson or shapefile format, covering specified periods. 
Additionally the script should calculate the normalized difference vegetation index (NDVI) 
statistics for the AOI and plot the change in vegetation over time on a graph. :

### Technologies and Tools Used

1. Python

2. Rasterio

3. SentinelHub

### Datasets

The dataset used is the Sentinel2 Level 2A Collection covering AOI of choice. 

### Files

Below are the folders and files associated with the completed exercise.

 > ndvi_pipeline.py - Script for task

 > ndvi_pipeline_script.py - Refined Script for task

 > ndvi_pipeline.ipynp - Jupyter Notebook Used to create script before conversion. 

 > README.md - Markdown file for description of the task.

 > requirements.txt - Text file with the necessary python libraries and packages to install with pip or conda.

### Execution

Preferable mode of execution is with Jupyter Notebooks.
The following steps should be followed in order to run the script.

The script is written with python and hence should be installed on your local
machine.

1. Open your Integrated Development Environment on your local machine if 
   that is your preferred mode of execution. If not, skip this step and 
   move to step 2.

2. Clone the Github repo on your local machine. 

3. Navigate to the ndvi-pipeline folder on your local machine.

	with command prompt ==> `cd /path/to/ndvi-pipeline` 

3. Create a virtual environment with conda or pip and activate it by using the commands below:
    
    To create a virtual environment using pip, you can use the following commands:

	```
	pip install virtualenvwrapper
	mkvirtualenv <name>
	workon <name>
    ```

    To create a virtual environment using conda, you can use the following command:

    `conda create --name myenv`

    Replace myenv with the name you want to give to your virtual environment.

    You can also specify the Python version to be used in the environment:

    `conda create --name myenv python=3.8`

    This command will create a virtual environment named myenv with Python version 3.8.

    After creating the virtual environment, you can activate it using:

    `conda activate myenv`

4. Install the necessary libraries from the requirements.txt file. 

    with conda ==> `conda install --file requirements.txt`

	with pip ==> `pip install -r /path/to/requirements.txt`


5. Open jupyter notebook on the command line by running the command below:

   `jupyter lab`

   Simply type this command in your terminal or command prompt and it will start Jupyter Lab in your default web browser.
   With jupyter notebook you just have to open the ndvi_pipeline.ipynb file and run the code cells.
   Pay attention to comments and run the cells accordingly. 

