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
                db_img_list = [misc.imread(os.path.expanduser(self.database + i)) for i in imagefiles]
                db_img_crop = []
                for db in db_img_list:
                    bounding_boxes, _ = align.detect_face.detect_face(
                        db, minsize, pnet,
                        rnet, onet, threshold, factor)
                    for (x1, y1, x2, y2, acc) in bounding_boxes:
                        w = x2-x1
                        h = y2-y1
                        y1c = int((y1+y2)/2.0 - max(w, h)/2.0)
                        y2c = int((y1+y2)/2.0 + max(w, h)/2.0)
                        x1c = int((x1+x2)/2.0 - max(w, h)/2.0)
                        x2c = int((x1+x2)/2.0 + max(w, h)/2.0)
                        cropped = db[y1c:y2c, x1c:x2c]
                        cropped = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_CUBIC)
                        db_img_crop.append(facenet.prewhiten(cropped))
                        break

                db_images = np.array(db_img_crop)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                feed_dict = { images_placeholder: db_images, phase_train_placeholder:False }
                db_emb = sess.run(embeddings, feed_dict=feed_dict)

                # nrof_images = len(imagefiles)
                # print('Images:')
                # for i in range(nrof_images):
                    # print('%1d: %s' % (i, imagefiles[i]))
                # print('')
                # print('Distance matrix')
                # print('    ')
                # for i in range(nrof_images):
                    # print '    %1d     ' % i,
                # print('')
                # for i in range(nrof_images):
                    # print '%1d  ' % i,
                    # for j in range(nrof_images):
                        # dist = np.sqrt(np.sum(np.square(np.subtract(db_emb[i,:], db_emb[j,:]))))
                        # print '  %1.4f  ' % dist,
                    # print('')

                while not rospy.is_shutdown():
                    img = rospy.wait_for_message("/usb_cam/image_raw/compressed", CompressedImage)
                    img_np_arr = np.fromstring(img.data, np.uint8)
                    encoded_img = cv2.imdecode(img_np_arr, 1)
                    flipped_img = cv2.flip(encoded_img, 1)
                    bounding_boxes, _ = align.detect_face.detect_face(
                        flipped_img, minsize, pnet,
                        rnet, onet, threshold, factor)

                    detected = []
                    textlocs = []
                    for (x1, y1, x2, y2, acc) in bounding_boxes:
                        w = x2-x1
                        h = y2-y1
                        y1c = int((y1+y2)/2.0 - max(w, h)/2.0)
                        y2c = int((y1+y2)/2.0 + max(w, h)/2.0)
                        x1c = int((x1+x2)/2.0 - max(w, h)/2.0)
                        x2c = int((x1+x2)/2.0 + max(w, h)/2.0)
                        if w <= 0 or h <= 0 or x1c < 0 or y1c < 0 or x2c > np.shape(flipped_img)[1] or y2c > np.shape(flipped_img)[0]:
                            continue
                        cv2.rectangle(flipped_img, (int(x1), int(y1)), (int(x1+w),
                                                      int(y1+h)), (255,0,0), 2)
                        cropped = flipped_img[y1c:y2c, x1c:x2c]
                        cropped = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_CUBIC)
                        detected.append(facenet.prewhiten(cropped))
                        textlocs.append((int(x1), int(y2)))
                    detected_imgs = np.array(detected)

                    if len(detected_imgs) > 0:
                        feed_dict = { images_placeholder: detected_imgs, phase_train_placeholder:False }
                        det_emb = sess.run(embeddings, feed_dict=feed_dict)
                        for det in range(len(detected_imgs)):
                            best = 10.0
                            bestdet = ''
                            for db in range(len(imagefiles)):
                                dist = np.sqrt(np.sum(np.square(np.subtract(db_emb[db,:], det_emb[det,:]))))
                                print 'distance to ' + imagefiles[db] + ': ' + str(dist)
                                if dist < best:
                                    best = dist
                                    bestdet = imagefiles[db]
                            if best < 0.9:
                                print 'best: ' + bestdet
                                cv2.putText(flipped_img, bestdet.split('_')[0], textlocs[det], cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,255),2,cv2.LINE_AA)
                            else:
                                cv2.putText(flipped_img, 'Unknown', textlocs[det], cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,255),2,cv2.LINE_AA)

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
