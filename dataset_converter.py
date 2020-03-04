import os

import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from pathlib import Path

import lxml.etree
from fastprogress import fastprogress

flags.DEFINE_string('data_dir', './data/images/', 'path to raw PASCAL VOC dataset')
flags.DEFINE_string('classes', './data/classes.txt', 'classes file')


def build_example(annotation, class_map):
    img_path = os.path.join(FLAGS.data_dir, annotation['folder'].replace(" – ", "-").replace("–", "-"), annotation['filename'])
    img_raw = open(img_path, 'rb').read()

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    views = []
    difficult_obj = []
    if 'object' in annotation:
        for obj in annotation['object']:
            difficult = bool(int(obj['difficult']))
            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_map[obj['name'].replace(" – ", "-").replace("–", "-")])
            truncated.append(int(obj['truncated']))
            views.append(obj['pose'].encode('utf8'))

    return tf.train.Example(features=tf.train.Features(feature={
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[annotation['filename'].encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
    }))


def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def process_split(split, class_map):
    writer = tf.io.TFRecordWriter(os.path.join(FLAGS.data_dir, f"military_{split}.tfrecord"))
    image_list = open(os.path.join(FLAGS.data_dir, 'data_%s.txt' % split), encoding='utf-8').read().splitlines()
    logging.info("Image list loaded: %d", len(image_list))
    for image in fastprogress.ProgressBar(image_list):
        annotation_xml = os.path.join("data", Path(image).with_suffix(".xml"))
        annotation_xml = lxml.etree.fromstring(open(annotation_xml).read())
        annotation = parse_xml(annotation_xml)['annotation']
        tf_example = build_example(annotation, class_map)
        writer.write(tf_example.SerializeToString())
    writer.close()


def main(_argv):
    class_map = {name: idx for idx, name in enumerate(
        open(FLAGS.classes, encoding='utf-8').read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)

    splits = ["train", "val"]

    for split in splits:
        logging.info(f"Processing {split} dataset")
        process_split(split, class_map)

    logging.info("Done")


if __name__ == '__main__':
    app.run(main)
