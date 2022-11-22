# Water Depth

The plugin estimates water depth from an image by recognizing measuring stick. The plugin automatically uses Nvidia GPU for inferencing when available. When you run the script, you have to provide the mapping of segmented pixel height and water depth.

## How to use

```bash
# -stream option is required to indicate source of a streaming input
# for example, python3 app.py -stream bottom
#   images from the camera named "bottom"

# to count water depth with a small portion of the input images to focus on the measuring stick (--cropping) and mapping information (--mapping)
$ python3 app.py --cropping "200 700 600 800" --mapping "469,6 467,7 465,8 463,9 461,10 459,11 457,12 455,13 453,14 450,15 448,16 446,17 444,18 442,19 440,20"

# to segment pixels as measuring stick which has values higher than 25 (the values are between (0, 255))
$ python3 app.py --threshold 25 --cropping "200 700 600 800" --mapping "469,6 467,7 465,8 463,9 461,10 459,11 457,12 455,13 453,14 450,15 448,16 446,17 444,18 442,19 440,20"

# to sample the image used in interencing
$ python3 app.py --sampling-interval 0 --cropping "200 700 600 800" --mapping "469,6 467,7 465,8 463,9 461,10 459,11 457,12 455,13 453,14 450,15 448,16 446,17 444,18 442,19 440,20"

# to estimate every 10 seconds
$ python3 app.py --continuous --interval 10 --cropping "200 700 600 800" --mapping "469,6 467,7 465,8 463,9 461,10 459,11 457,12 455,13 453,14 450,15 448,16 446,17 444,18 442,19 440,20"
```

## funding
[NSF 1935984](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1935984)

## collaborators
Bhupendra Raut, Dario Dematties Reyes, Joseph Swantek, Neal Conrad, Nicola Ferrier, Pete Beckman, Raj Sankaran, Robert Jackson, Scott Collis, Sean Shahkarami, Seongha Park, Sergey Shemyakin, Wolfgang Gerlach, Yongho Kim

## Acknowledgement
We thank our many collaborators, including site PIs and technicians, for their efforts in support of PhenoCam. The development of PhenoCam has been funded by the Northeastern States Research Cooperative, NSF's Macrosystems Biology program (awards EF-1065029 and EF-1702697), and DOE's Regional and Global Climate Modeling program (award DE-SC0016011). We acknowledge additional support from the US National Park Service Inventory and Monitoring Program and the USA National Phenology Network (grant number G10AP00129 from the United States Geological Survey), and from the USA National Phenology Network and North Central Climate Science Center (cooperative agreement number G16AC00224 from the United States Geological Survey). Additional funding, through the National Science Foundation's LTER program, has supported research at Harvard Forest (DEB-1237491) and Bartlett Experimental Forest (DEB-1114804). We also thank the USDA Forest Service Air Resource Management program and the National Park Service Air Resources program for contributing their camera imagery to the PhenoCam archive.
