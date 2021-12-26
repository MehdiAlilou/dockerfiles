Brain Tumor detection and Segmentation model

This project uses maskRcnn deep model to detect and segment brain tumors in brain MR images.

docker build -t maskrcnn_image .
docker run -v C:\matcode\data\brainTumor\Br35H-Mask-RCNN\:/app/input_output maskrcnn_image