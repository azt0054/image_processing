import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))

#-- image directories --#
image_root_dir = os.path.join(dir_path,"images")
s3_image_local_dir = os.path.join(image_root_dir,"s3")
source_image_dir = os.path.join(image_root_dir,"s3")
processed_image_dir = os.path.join(image_root_dir,"processed")
background_image_dir = os.path.join(processed_image_dir,"background_images")
backgroundless_image_dir = os.path.join(processed_image_dir,"backgroundless_images")

s3_bucket_name = "shipt-staging"
s3_image_dir = "images/products/"

max_download_count = 100

acceptable_image_type = ["jpg", "jpeg", "png"]

corner_size = 8                                                   #for background detection
edge_depth = 2                                                     #for background detection
kernel_size = (5,5)                                                #for pixel level operations

output_image_size = [500,500]

gamma_limit = 0.2

threshold_block_size = 171                                         #grayscale limit
min_edge_threshold_limit = 30                                      #edge detection limit
max_edge_threshold_limit = 90                                      #edge detection limit

min_contour_size = 500                                             #contour filtering limit
min_contour_distance = 0                                           #contour filtering limit

max_contour_size = 2000                                            #contour filtering limit
max_contour_distance = 50                                          #contour filtering limit

background_color = 255                                             #(0-255)
resize_margin = 20                                                 #margin for image after resize

grabcut_limit = 2                                                  #number of iteration for filtering

wait_time=2000                                                        #display time in milliseconds(0 for infintite)
