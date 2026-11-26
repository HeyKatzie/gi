<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# RoadCount AI: Vehicle Classification for Rural Roads

Building AI course project

## Summary

RoadCount AI uses computer vision to automatically count and classify vehicles (cars, buses, trucks, motorcycles) from roadside video. It aims to replace slow manual traffic counts on rural roads, starting with the VRAEM region in Peru.


## Background

Road design and maintenance depend on knowing how many vehicles use a road and what types they are. Heavy trucks, for example, damage pavement much more than cars, so the vehicle mix directly affects pavement design. In Peru, traffic studies usually rely on manual counts: people stand by the road for several days, writing down every vehicle that passes.

This approach has several problems:
* manual counts are expensive, tiring and prone to human error, especially at night or during long shifts
* in remote rural areas such as the VRAEM (the valley of the Apurímac, Ene and Mantaro rivers), access is difficult and counting teams face safety risks
* counts are done only for a few days, so seasonal changes in traffic are often missed
* without reliable traffic data, investment decisions for rural roads are harder to justify

As a civil engineering student, I have seen how important traffic data is for road projects, and how hard it is to get in rural areas. This idea is also connected to my final degree project on vehicle classification for rural roads in the VRAEM.


## How is it used?

1. A camera is installed next to the road (for example on a pole), or existing video footage of the road is used.
2. The video is processed by the system, which detects each vehicle, follows it across frames so it is counted only once, and assigns it a class.
3. The results are saved as a table with the number of vehicles per class and per hour.
4. Engineers use the table to estimate traffic volumes and the vehicle mix needed for road and pavement design.

The main users are road engineers, consultants and public agencies that plan and maintain rural roads. Important needs to take into account:
* the system must work with low-cost cameras and limited power supply
* rural sites often have no internet connection, so the video may need to be processed offline later
* it must handle dust, rain, poor lighting and unpaved roads
* the vehicle classes should match the categories used in national traffic studies

<!-- Add an image here once you upload one to your repository, for example:
![Vehicle detection example](images/detection_example.png) -->

A simplified example of how the counting could work, using a pre-trained object detection model:

```
from ultralytics import YOLO

model = YOLO("yolov8n.pt")              # pre-trained object detection model
vehicle_classes = ["car", "motorcycle", "bus", "truck"]
counted_ids = set()
counts = {c: 0 for c in vehicle_classes}

# track objects across frames so each vehicle is counted only once
for result in model.track(source="road_video.mp4", stream=True, persist=True):
    if result.boxes.id is None:
        continue
    for box, track_id in zip(result.boxes, result.boxes.id):
        label = model.names[int(box.cls)]
        if label in vehicle_classes and int(track_id) not in counted_ids:
            counted_ids.add(int(track_id))
            counts[label] += 1

print(counts)
```


## Data sources and AI methods

The system needs labeled images of vehicles, where each vehicle has a bounding box and a class.

| Data source | Use |
| ----------- | ----------- |
| Public datasets such as [COCO](https://cocodataset.org/) | Pre-training a general vehicle detector |
| Own video recordings of rural roads in the VRAEM | Fine-tuning and testing in real local conditions |
| Manual counts done at the same time as the recordings | Checking the accuracy of the system |

AI methods:
* **Object detection** with a convolutional neural network (for example the YOLO family of models) to find vehicles in each frame
* **Supervised learning**: the model is fine-tuned with local images labeled by hand, so it learns local vehicle types such as mototaxis and small trucks
* **Object tracking** to follow each vehicle across frames and avoid counting it twice
* **Evaluation** with separate training and test sets, comparing the automatic counts with manual counts


## Challenges

* The project does not measure vehicle weight or axle loads, which are also needed for pavement design.
* Accuracy may drop at night, in heavy rain or dust, or when vehicles block each other.
* Local vehicle types (mototaxis, modified trucks) are rare in public datasets, so local data must be collected and labeled, which takes time.
* Ethical and privacy considerations: cameras may capture people and license plates. The system should only store counts, not identify people or vehicles, and recordings should be deleted or anonymized after processing.
* Installing equipment in remote areas has security and maintenance costs.


## What next?

* Collect and label a larger local dataset of Peruvian rural roads
* Run the model on low-cost edge devices so counting can happen directly at the roadside
* Add speed estimation and peak-hour analysis
* Connect the results to road planning tools used by engineers and public agencies

To move forward, I would need help with data labeling, access to field recording sites, and guidance from people with experience in deploying computer vision systems.


## Acknowledgments

* Inspired by the Building AI course by Reaktor Innovations and the University of Helsinki
* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) / [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) for object detection and tracking
* [COCO dataset](https://cocodataset.org/) / [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) for pre-trained vehicle classes
* My final degree project in civil engineering at UTEC (Universidad de Ingeniería y Tecnología), Peru
