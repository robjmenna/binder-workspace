import cv2
import os
import argparse
from google.protobuf import text_format
import csv
from object_detection.utils import dataset_util
import tensorflow as tf
import zipfile
import object_detection.protos.string_int_label_map_pb2
import urllib.request

def create_record(imgfile, boxes=None):
    img = cv2.imread(imgfile)
    _, encoded_image_data = cv2.imencode(".jpg", img)
    height, width, _ = img.shape
    image_format = b"jpeg"
    _, filename = os.path.split(imgfile)

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    if boxes is None:
        boxes = []

    for box in boxes:
        xmins.append(box[0] / width)
        ymins.append(box[1] / height)
        xmaxs.append(box[2] / width)
        ymaxs.append(box[3] / height)
        classes.append(box[4])
        classes_text.append(box[5])

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(filename.encode()),
                "image/source_id": dataset_util.bytes_feature(filename.encode()),
                "image/encoded": dataset_util.bytes_feature(
                    encoded_image_data.tobytes()
                ),
                "image/format": dataset_util.bytes_feature(image_format),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(
                    classes_text
                ),
                "image/object/class/label": dataset_util.int64_list_feature(classes),
            }
        )
    )
    return tf_example

def main(args):
    # open the readme and read out the lines that have the 
    # class labels
    readmefile = os.path.join(args.data_dir, "ReadMe.txt")
    with open(readmefile) as rm:
        rawclasses = list(rm)[39:82]

    # build the proto label map
    label_list = object_detection.protos.string_int_label_map_pb2.StringIntLabelMap()
    label_dict = {}
    for line in rawclasses:
        item = object_detection.protos.string_int_label_map_pb2.StringIntLabelMapItem()
        id, name = line.split("=")
        item.id = int(id) + 1
        item.name = name.strip()
        label_list.item.append(item)
        label_dict[item.id] = item.name

    labelmapout = os.path.join(args.output_dir, "label_map.pbtxt")
    with open(labelmapout, "w") as file:
        st = text_format.MessageToString(label_list)
        file.write(st)

    # open and read the ground truth data
    gtfile = os.path.join(args.data_dir, "gt.txt")
    imagedict = {}
    with open(gtfile) as file:
        for line in file:
            columns = line.split(';')
            imgfile = f"{os.path.join(args.data_dir, columns[0])}"
            xmin, ymin, xmax, ymax = (int(x) for x in columns[1:5])
            id = int(columns[5]) + 1
            if imgfile not in imagedict:
                imagedict[imgfile] = []
            imagedict[imgfile].append((xmin, ymin, xmax, ymax, id, label_dict[id].encode()))

    # write the training data to its own file
    records = [os.path.join('data', 'images', x) for x in os.listdir('data\\images') if x.endswith('.ppm')]
    trainoutput = os.path.join(args.output_dir, "train.record")
    with tf.io.TFRecordWriter(trainoutput) as writer:
        for imgfile in records[:600]:
            if imgfile in imagedict:
                boxes = imagedict[imgfile]
                example = create_record(imgfile, boxes)
            else:
                example = create_record(imgfile)
            writer.write(example.SerializeToString())

    # write the test data to its own file
    testoutput = os.path.join(args.output_dir, "test.record")
    with tf.io.TFRecordWriter(testoutput) as writer:
        for imgfile in records[600:]:
            if imgfile in imagedict:
                boxes = imagedict[imgfile]
                example = create_record(imgfile, boxes)
            else:
                example = create_record(imgfile)
            writer.write(example.SerializeToString())

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Prepare the GTRSB dataset for the Tensorflow Object Detection API."
    )
    parser.add_argument(
        "data_dir",
        help="the directory that holds the full GTRSB dataset including ReadMe.txt and gt.txt",
    )
    parser.add_argument(
        "--output_dir",
        default=".\\",
        help="where to save the pbtxt and record files. If not specified the files will be saved to the current directory",
    )

    args = parser.parse_args()
    main(args)    
