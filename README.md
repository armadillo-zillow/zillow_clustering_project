# Zillow Clustering Project

## Purpose
This repository holds all resources used in the attainment of the goals established for the Zillow Clustering Project.

## Goal
Improve our original estimate of the log error by using clustering methodologies.

## Data
2017 properties and prediction data from zillow database for single unit/single family homes.

## Audience
The audience for this project are our peers and superiors on the data science team. Given the baseline knowledge of the audience, the content in the final_project.ipynb file contains more technical jargon and minutia than may be present in a report or presentation to senior leadership or other non-SMEs.

## Deliverables

### Need to Haves
1. Final Report (jupyter notebook)
2. Modules:
    - wrangle_zillow.py
    - model.py
3. README.md

### Nice to Haves:
1. Modules:
    - preprocessing.py
    - explore.py
2. Explore logerror for rows with high percentage of missing values
    - If logerror for a particular feature is low it could indicate the value of that information
    - Feature describing amount of data missing
3. Explore dataset with properties added that do not have latitude or longitude
4. Subset by regionidneighborhood for modeling

## Data Dictionary
A data dictionary for the features contained in our final report can be found [here](https://rstudio-pubs-static.s3.amazonaws.com/321635_482e51c0348b4d01a7d3ed6cf86eb2ae.html).

## Cloning & Reproducibility
All files necessary for cloning and reproducing the work found in the final_project.ipynb file are contained within this repository.

## Planning
Additional planning documentation can be found [here](https://docs.google.com/document/d/1MtDXR6I8l17Uzs2W4m7RgUuDnVxEzJ0aZqfjWwlpc4M/edit?usp=sharing).