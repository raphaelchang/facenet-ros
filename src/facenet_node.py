#!/usr/bin/env python

import rospy
import facenet
import align.detect_face
import tensorflow as tf
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
import os
from scipy import misc

NODE_NAME = "facenet"

class FaceNetNode(object):

    def __init__(self, model, database):
        self.model = model
        self.database = database

    def run(self):
        minsize = 20 # minimum size of face
        threshold = [ 0.7, 0.8, 0.8 ]  # three steps's threshold
        factor = 0.709 # scale factor

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

	output_publisher = rospy.Publisher('facenet/image_raw/compressed', CompressedImage, queue_size=1)

        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model(self.model)
                imagefiles = os.listdir(self.database)
                db_images = np.array([misc.imread(os.path.expanduser(self.database + i)) for i in imagefiles])
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                feed_dict = { images_placeholder: db_images, phase_train_placeholder:False }
                db_emb = sess.run(embeddings, feed_dict=feed_dict)

                while not rospy.is_shutdown():
                    img = rospy.wait_for_message("/usb_cam/image_raw/compressed", CompressedImage)
                    img_np_arr = np.fromstring(img.data, np.uint8)
                    encoded_img = cv2.imdecode(img_np_arr, 1)
                    flipped_img = cv2.flip(encoded_img, 1)
                    bounding_boxes, _ = align.detect_face.detect_face(
                        flipped_img, minsize, pnet,
                        rnet, onet, threshold, factor)

                    detected = []
                    for (x1, y1, x2, y2, acc) in bounding_boxes:
                        w = x2-x1
                        h = y2-y1
                        cv2.rectangle(flipped_img, (int(x1), int(y1)), (int(x1+w),
                                                      int(y1+h)), (255,0,0), 2)
                        cropped = flipped_img[int((y1+y2)/2.0 - max(w, h)/2.0):int((y1+y2)/2.0 + max(w, h)/2.0), int((x1+x2)/2.0 - max(w, h)/2.0):int((x1+x2)/2.0 + max(w, h)/2.0)]
                        cropped = misc.imresize(cropped, (160, 160), interp='bilinear')
                        detected.append(cropped)
                    detected_imgs = np.array(detected)

                    if len(detected_imgs) > 0:
                        feed_dict = { images_placeholder: detected_imgs, phase_train_placeholder:False }
                        det_emb = sess.run(embeddings, feed_dict=feed_dict)
                        best = 10.0
                        bestdet = ''
                        for det in range(len(detected_imgs)):
                            for db in range(len(imagefiles)):
                                dist = np.sqrt(np.sum(np.square(np.subtract(db_emb[db,:], det_emb[det,:]))))
                                print 'distance to ' + imagefiles[db] + ': ' + str(dist)
                                if dist < best:
                                    best = dist
                                    bestdet = imagefiles[db]

                        print 'best: ' + bestdet

                    msg = CompressedImage()
                    msg.header.stamp = rospy.Time.now()
                    msg.format = "jpeg"
                    msg.data = np.array(cv2.imencode('.jpg', flipped_img)[1]).tostring()
                    output_publisher.publish(msg)


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=False)
    model = rospy.get_param("~model")
    database = rospy.get_param("~database")
    node = FaceNetNode(model, database)
    node.run()
