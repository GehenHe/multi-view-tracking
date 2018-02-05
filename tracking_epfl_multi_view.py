# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import, print_function

import argparse
import os
import time
import cv2
import numpy as np
import sys
this_dir = os.path.realpath(__file__).rsplit('/', 1)[0]
extract_path = os.path.join(this_dir, 'reID_caffe')
sys.path.append(this_dir)
sys.path.append(extract_path)

from seq_info import *
from extract_per_frame import load_ReID_net, extract
from preprocess.visualization import draw_tracker
from preprocess.homography_matrix import terrace_H
from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tf_faster_rcnn.pedestiran_det import PedestrianDet
import copy
from glob import glob
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from operator import xor
import Queue


def create_det_dict( detection_file):
    det_file = open(detection_file, 'r')
    det_list = det_file.readlines()
    detect_dict = {}
    for det in det_list:
        det = det.strip().split(',')
        frame_id = int(det[0])
        bbox = map(float, det[2:6])
        bbox = map(int, bbox)
        # background = int(det[6])
        # non_ped = int(det[7])
        score = float(det[-1])
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], score]
        detect_dict.setdefault(frame_id, []).append(bbox)
    return detect_dict


def view_to_top(H,xy):
    temp = H*np.mat([xy[0],xy[1],1]).T
    temp = temp/temp[2]
    return temp[0:2]


class TrackObject:
    def __init__(self,trackers=None,detector_name='coco-res101',min_confidence=0.8, detection_file = None,  nms_max_overlap=1.0, min_detection_height=0,
                 max_cosine_distance=0.2, nn_budget=100, view_num = 4,display=False):
        self.detection_file= detection_file
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.min_detection_height = min_detection_height
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.display = display
        self.caffeReIDNet = load_ReID_net()
        # self.detector = PedestrianDet(detector_name)

        # global track id
        self.pre_multi_next_id = [1]
        self.multi_next_id = [1]

        # map view_id to global id
        self.pre_id_map = {i:{} for i in range(view_num)}
        self.id_map = {i:{} for i in range(view_num)}

        pre_trackers = []
        for i in range(view_num):
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", matching_threshold=0.4, budget=100)
            pre_trackers.append(Tracker(metric,multi_next_id=self.pre_multi_next_id))
        self.pre_multi_trackers = pre_trackers
        self.multi_trackers = copy.deepcopy(self.pre_multi_trackers)

        # save tracking result by {}[view][frame_idx]
        self.pre_tracking_results = [{} for i in range(len(pre_trackers))]
        self.tracking_results = copy.deepcopy(self.pre_tracking_results)


    def create_detections(self, detection_mat, frame_idx, min_height=0):
        """Create detections for given frame index from the raw detection matrix.

        Parameters
        ----------
        detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
        frame_idx : int
        The frame index.
        min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

        Returns
        -------
        List[tracker.Detection]
        Returns detection responses at given frame index.

        """
        detection_list = []
        if len(detection_mat) != 0:
            frame_indices = detection_mat[:, 0].astype(np.int)
            mask = frame_indices == frame_idx
            for row in detection_mat[mask]:
                bbox, confidence, feature = row[2:6], row[6], row[10:]
                if bbox[3] < min_height:
                    continue
                if bbox[3]<=bbox[2]:
                    continue
                detection_list.append(Detection(bbox, confidence, feature))
        return detection_list

    def multi_view_detection(self,images):
        detection_list = []
        for view_idx,image in enumerate(images):
            det_frRCNN = self.detector.det_person(image)
            if det_frRCNN is not None:
                idx = np.where(det_frRCNN[:, -1] >= 0.1)[0]
                det_frRCNN = det_frRCNN[idx, :]
                detections = np.asarray(det_frRCNN)
            else:
                detections = np.reshape(np.asarray([]),[0,5])
            detection_list.append(detections)
        return detection_list

    def multi_view_prematching(self,images,detection_list,H):
        self.pre_multi_trackers = [self.multi_trackers[i]._copy() for i in range(len(self.multi_trackers))]
        self.pre_tracking_results = copy.deepcopy(self.tracking_results)
        for view_idx,det_frRCNN in enumerate(detection_list):
            image = images[view_idx]
            if len(det_frRCNN)>0:
                det_fea = extract(self.caffeReIDNet, image, det_frRCNN, frame_idx)
                extr_end = time.time()
                # print("(2) extract feature time: %.2f"%(extr_end-extr_start))
                detections = self.create_detections(
                    det_fea, frame_idx, self.min_detection_height)
                self.pre_multi_trackers[view_idx].predict()
                self.pre_multi_trackers[view_idx].update(detections,self.pre_multi_next_id)
            else:
                self.pre_multi_trackers[view_idx].predict()

        for index,tracker_view in enumerate(tracker.pre_multi_trackers):
            if len(tracker_view.tracks)<=0:
                self.pre_tracking_results[index].setdefault(frame_idx, {})
            for track in tracker_view.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    self.pre_tracking_results[index].setdefault(frame_idx, {})
                    continue
                bbox = track.to_tlwh()
                x = int(bbox[0]+bbox[2]/2)
                y = int(bbox[1]+bbox[3])
                new_xy = view_to_top(H[index],[x,y])
                self.pre_tracking_results[index].setdefault(frame_idx,{}).setdefault(track.track_id,new_xy)

    def multi_view_matching(self,images,detection_list,H):
        for view_idx,det_frRCNN in enumerate(detection_list):
            image = images[view_idx]
            if det_frRCNN is not None:
                idx = np.where(det_frRCNN[:, -1] >= self.min_confidence)[0]
                det_frRCNN = det_frRCNN[idx, :]
                det_fea = extract(self.caffeReIDNet, image, det_frRCNN, frame_idx)
                detections = self.create_detections(
                    det_fea, frame_idx, self.min_detection_height)
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
                detections = [detections[i] for i in indices]
                self.multi_trackers[view_idx].predict()
                self.multi_trackers[view_idx].update(detections,self.multi_next_id)
            else:
                self.multi_trackers[view_idx].predict()

        for index,tracker_view in enumerate(tracker.multi_trackers):
            if len(tracker_view.tracks)<=0:
                self.pre_tracking_results[index].setdefault(frame_idx, {})
            for track in tracker_view.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    self.pre_tracking_results[index].setdefault(frame_idx, {})
                    continue
                bbox = track.to_tlwh()
                x = int(bbox[0]+bbox[2]/2)
                y = int(bbox[1]+bbox[3])
                new_xy = view_to_top(H[index],[x,y])
                self.tracking_results[index].setdefault(frame_idx,{}).setdefault(track.track_id,new_xy)

    def draw_trackers(self,frame_idx,images,is_det=False,detections=None):
        view_num = len(images)
        row_num = np.sqrt(view_num)
        show_list = []
        for view,image in enumerate(images):
            image_show = draw_tracker(image,self.multi_trackers[view].tracks,self.id_map[view])
            image_show = cv2.cvtColor(image_show,cv2.COLOR_BGR2RGB)
            if is_det:
                det = detections[view]
                for bbox in det:
                    image_show = cv2.rectangle(image_show,(bbox[0],bbox[1]),(bbox[2],bbox[3]),[0,0,0],1)
            show_list.append(image_show)
        for i in range(view_num):
            image = show_list[i]
            plt.subplot(row_num,view_num/row_num,i)
            plt.axis('off')
            plt.imshow(image)
        plt.savefig('/home/gehen/PycharmProjects/multi_view_tracking/output/vis_temp/{}.png'.format(frame_idx))

    def generate_sequence(self,tracking_results=None,multi_trackers=None,queue_length=10):
        frame_idx = max(self.pre_tracking_results[0].keys())
        start_frame = frame_idx-queue_length
        frame_list = range(start_frame+1,frame_idx+1)
        data = []
        pre_tracks_list = []
        for view_id,tracker in enumerate(tracking_results):
            view_data = []
            id_list = tracker[frame_idx].keys()
            id_list.sort()
            for id in id_list:
                id_data = []
                for frame in frame_list:
                    try:
                        if id in tracker[frame].keys():
                            id_data.append(tracker[frame][id])
                        else:
                            id_data.append([])
                    except:
                        id_data.append([])
                temp = [track.track_id for track in multi_trackers[view_id].tracks]
                temp_index = temp.index(id)
                track = multi_trackers[view_id].tracks[temp_index]
                if track.is_confirmed() and track.time_since_update <= 1:
                    pre_tracks_list.append(track)
                view_data.append(id_data)
            data.append(view_data)
        return data,pre_tracks_list

    def dist_in_seq(self,seq1,seq2,thresh=50):
        dist = 0
        count = 0
        assert len(seq1)==len(seq2),"sequence length not equal!"
        for i in range(len(seq1)):
            if isinstance(seq1[i],list) and isinstance(seq2[i],list):
                continue
            elif xor(not isinstance(seq1[i], list),not isinstance(seq2[i], list)):
                dist = dist+thresh*2*np.exp(-(len(seq1)-1-i)*0.1)
                count = count+1
            elif not isinstance(seq1[i], list) and not isinstance(seq2[i], list):
                dist = dist+euclidean(seq1[i],seq2[i])*np.exp(-(len(seq1)-1-i)*0.1)
                count = count+1
        return dist/count

    def point_dist(self,point1,point2):
        x_dist = np.linalg.norm(point1[0]-point2[0])
        y_dist = np.linalg.norm(point2[1]-point2[1])
        dist = np.linalg.norm(np.sqrt(2)*x_dist+(1/np.sqrt(2))*y_dist)
        return dist

    def cal_distance(self,data):
        seq_list = []
        for view_data in data:
            for id_data in view_data:
                seq_list.append(id_data)
        num = len(seq_list)
        dist_matrix = np.mat(np.zeros([num,num]))
        for i in range(num):
            for j in range(i+1,num):
                dist_matrix[i,j] = self.dist_in_seq(seq_list[i],seq_list[j])
        person_num = [len(data[i]) for i in range(len(data))]
        return dist_matrix+dist_matrix.T,person_num

    def assignment_matrix(self,dist_matrix,num_list):
        assign_matrix = np.zeros(dist_matrix.shape)
        for i in range(dist_matrix.shape[0]):
            count = 0
            for view,num in enumerate(num_list):
                if num <1:
                    continue
                index = np.where(dist_matrix[i]==np.min(dist_matrix[i,count:count+num]))
                if dist_matrix[i,index[1]]<=10:
                    assign_matrix[i,index[1]] = 1
                count+=num
        return assign_matrix

    def ID_match(self,assign_matrix,num_list,tracker_list,id_map,_next_id):
        # tracker_list = []
        # for track in [track for view_tracker in self.pre_multi_trackers for track in view_tracker.tracks]:
        #     if not track.is_confirmed() or track.time_since_update > 1:
        #         continue
        #     tracker_list.append(track)

        assert len(tracker_list)==sum(num_list),'tracker length is {}, assign matrix is {}'.format(len(tracker_list),sum(num_list))
        if len(tracker_list)<2:
            return
        temp_matrix = np.unique(assign_matrix,axis=0)
        for i in range(temp_matrix.shape[0]):
            item = temp_matrix[i,:]
            index_list = np.where(item==1)[0]
            for index in index_list:
                id_map[self.search_view(index,num_list)].setdefault(tracker_list[index].track_id,tracker_list[index].track_id)
            id = min([id_map[self.search_view(index,num_list)][tracker_list[index].track_id] for index in index_list])
            for index in index_list:
                view = self.search_view(index,num_list)
                ori_id = tracker_list[index].track_id
                id_map.setdefault(view,{})
                id_map[view][ori_id] = id
        temp = max([person_id  for view_id in id_map for person_id in id_map[view_id]])
        _next_id[0] = temp+1



    def search_view(self,index,num_list):
        index = index+1
        for i in range(len(num_list)):
            if index<=sum(num_list[0:i+1]):
                return i






if __name__ == "__main__":
    video_list = os.listdir('/home/gehen/Downloads/images/tracking/EPFL-video/images')
    video_list.sort()
    video_list = ['terrace1-c0','terrace1-c1','terrace1-c2','terrace1-c3']
    video_dir = '/home/gehen/PycharmProjects/multi_view_tracking/data/EPFL/{}/img1'
    save_dir = '/home/gehen/PycharmProjects/multi_view_tracking/output/vis_temp'
    gt_dir = '/home/gehen/PycharmProjects/multi_view_tracking/data/EPFL'
    vis_dir = '/home/gehen/PycharmProjects/multi_view_tracking/output/vis_temp'

    image_dir = [video_dir.format(video) for video in video_list]
    img_list = os.listdir(image_dir[0])
    img_list.sort()
    tracker_list = []
    for i in range(len(image_dir)):
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", matching_threshold=0.4, budget=100)
        tracker_list.append(Tracker(metric))
    tracker = TrackObject(tracker_list,detector_name='coco-res101',min_confidence=0.98,min_detection_height=20,view_num=len(image_dir))
    past = time.time()

    result_list = [[] for i in range(len(image_dir))]
    person_dict_list = [{} for i in range(len(image_dir))]
    detect_dict_list = [create_det_dict(os.path.join(gt_dir,video_name,'gt/gt.txt')) for video_name in video_list]
    detection_result = {}

    #  get bboxs from file
    with open('terrace1.pkl','rb') as f:
        detection_result = pickle.load(f)

    # with open('/home/gehen/PycharmProjects/multi_view_tracking/data/detection/terrace1-c0.pkl','rb') as f:
    #     detection_result = pickle.load(f)

    for frame_idx,img in enumerate(img_list[900:1100]):
        frame_idx = int(img.split('.')[0])
        if frame_idx%100 == 0:
            now = time.time()
            print ('finish {:d}/{:d} , took {:.3f}s for each image'.format(int(frame_idx/100),int(len(img_list)/100),(now-past)/100.0))
            past = now
        images = [cv2.imread(os.path.join(image_dir[i],img)) for i in range(len(image_dir))]

        # detections = tracker.multi_view_detection(images)
        # detection_result.setdefault(frame_idx,detections)

        detections = detection_result[frame_idx]
        if frame_idx == 414:
            pass
        for i in range(len(detections)):
            det = detections[i]
            if det is not None and len(det)>0:
                index = np.where(((det[:,3]-det[:,1])>=min_height) & ((det[:,2]-det[:,0])>=min_width)&(det[:,-1]>=0.98))
                detections[i] = detections[i][index]
        tracker.multi_view_prematching(images,detections,terrace_H())
        dist_data,pre_tracks_list = tracker.generate_sequence(tracker.pre_tracking_results,tracker.pre_multi_trackers)
        dist_matrix,num_list = tracker.cal_distance(dist_data)
        assign_matrix = tracker.assignment_matrix(dist_matrix,num_list)
        tracker.ID_match(assign_matrix,num_list,pre_tracks_list,tracker.pre_id_map,tracker.pre_multi_next_id)

        # tracker.assignment_matrix(dist_matrix,num_list)

        tracker.multi_view_matching(images, detections,terrace_H())
        tracker.ID_match(assign_matrix, num_list, pre_tracks_list,tracker.id_map,tracker.multi_next_id)
        tracker.draw_trackers(frame_idx, images,True,detections)

        # pass
        # tracker.multi_view_prematching(images,detections,terrace_H())
        # tracker.draw_trackers(frame_idx,images)
        # dist_data = tracker.generate_sequence()
        # dist_matrix = tracker.cal_distance(dist_data)
        # det_gt = []
        # for i in range(4):
        #     try:
        #         det_gt.append(np.asarray(detect_dict_list[i][frame_idx]))
        #     except:
        #         det_gt.append(np.asarray([]))
        # det_gt = det_gt

    #     for index,tracker_view in enumerate(tracker.pre_multi_trackers):
    #         for track in tracker_view.tracks:
    #             if not track.is_confirmed() or track.time_since_update > 1:
    #                 continue
    #             bbox = track.to_tlwh()
    #             result_list[index].append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
    #             person_dict_list[index].setdefault(track.track_id,{}).setdefault(frame_idx,[bbox[0], bbox[1], bbox[2], bbox[3]])
    # # with open('terrace1.pkl','wb') as f:
    # #     pickle.dump(detection_result,f)
    # for idx,video in enumerate(video_list):
    #     trk_save_path = os.path.join(save_dir, video + '.txt')
    #     f = open(trk_save_path, 'w')
    #     result = result_list[idx]
    #     for row in result:
    #         print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (row[0], row[1], row[2], row[3], row[4], row[5]), file=f)
