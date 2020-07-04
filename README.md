# Social Distancing

A Social Distance Monitoring Tool using OpenCV.

## Outline

Firstly humans are identified in a video stream using YOLO(You Only Look Once)- a special kind of Convolutional Neural Network. 

Each person detected is bounded by a rectangular box in order to locate the person.

The camera perspective is transformed to a bird-eye view (top down) for effectively computing euclidean distance between people.

This part has to be configured manually. For more details please refer the 1st reference mentioned below.

The surrounding boxes are classified as follows-

1) Red-High Risk

2) Yellow-Medium Risk

3) Green-Low Risk

Another aspect incorporated was tracking people. This was accomplished using SORT(Simple Online Realtime Tracking) technique.

Each person is assigned a unique identity which is carried forward in subsequent frames.

This can be useful for further statistical analysis and computing violation metrics.

## References

1) Social Distancing-https://github.com/deepak112/Social-Distancing-AI

2) People Tracking-https://github.com/abewley/sort

3) Monitoring COVID-19 social distancing with person detection and tracking via fine-tuned YOLO v3 and Deepsort techniques-https://arxiv.org/abs/2005.01385




 
