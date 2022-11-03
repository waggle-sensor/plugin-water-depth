# Science

Monitoring water depth changes using image or video data in real-time is one of the most critical factors for forecasting and analyzing potential flooding in local stream, river, or other water reservoirs when there are strong rain storms, rapid snow melting, or hurricanes. It is challenging to come up with a robust water depth estimation method when we segment water area because shapes of water are changing rapidly, therefore we leveraged object-based image analysis, which used image segmentation method for segmenting measuring stick that is located in the stream has been adopted to estimate the water depth. 

# AI at Edge

The code runs the U-Net model which is trained with a Dataset that was collected from one of National Ecological Observatory Network (NEON) sites with a given time interval. In each run, it takes a still image from a given camera (bottom), and the model outputs measuring stick segmentation result, then the postprocessing algorithm maps the lowest pixel of the segmentation result to the depth of the water. The model cropped the images and resizes the image to 300x300 (WxH) as the model was trained with the size.

# Ontology

The code publishes measurement with topic `env.water.depth`. Value for a topic indicates the water depth in cm.

# Inference from Sage codes
To query the output from the plugin, you can do with python library 'sage_data_client':
```
import sage_data_client

# query and load data into pandas data frame
df = sage_data_client.query(
    start="-1h",
    filter={
        "name": "env.water.depth",
    }
)

# print results in data frame
print(df)
# print results by its name
print(df.name.value_counts())
# print filter names
print(df.name.unique())
```
For more information, please see [Access and use data documentation](https://docs.sagecontinuum.org/docs/tutorials/accessing-data) and [sage_data_client](https://pypi.org/project/sage-data-client/).
