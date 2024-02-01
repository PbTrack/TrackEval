import json
import os
import csv
import configparser
import numpy as np
import random
from ._base_dataset import _BaseDataset
from .. import utils
from .. import _timing
from ..utils import TrackEvalException

class SoccerNetGS(_BaseDataset):
    """Dataset class for the SoccerNet Challenge Game State (GS) task"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/SoccerNetGS/val'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/SoccerNetGS/'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'SPLIT_TO_EVAL': 'val',  # Valid: 'train', 'val', 'test', 'challenge'
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': True,  # Whether to print current config
            'DO_PREPROC': True,  # Whether to perform preprocessing (never done for MOT15)
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
            'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
            'SEQ_INFO': None,  # If not None, directly specify sequences to eval and their number of timesteps
            'GT_LOC_FORMAT': '{gt_folder}/{seq}/Labels-GameState.json',  # '{gt_folder}/{seq}/gt/gt.json'
            'SKIP_SPLIT_FOL': False,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
                                      # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
                                      # If True, then the middle 'benchmark-split' folder is skipped for both.
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())

        gt_set = self.config['SPLIT_TO_EVAL']
        self.gt_set = self.benchmark + '-' + self.config['SPLIT_TO_EVAL']
        if not self.config['SKIP_SPLIT_FOL']:
            split_fol = gt_set
        else:
            split_fol = ''
        self.gt_fol = os.path.join(self.config['GT_FOLDER'], split_fol)
        self.tracker_fol = os.path.join(self.config['TRACKERS_FOLDER'], split_fol)
        self.seq_list, self.seq_lengths = self._get_seq_info()
        self.eval_mode = 'distance' # 'distance' or 'classes'  # TODO
        self.all_classes = {}
        if self.eval_mode == 'classes':
            self.all_classes = extract_all_classes(self.config, self.gt_fol, self.seq_list)
            self.class_name_to_class_id = {clazz["name"]: clazz["id"] for clazz in self.all_classes.values()}
        else:
            self.class_name_to_class_id = {
                "person": 1,
            }
        self.should_classes_combine = True
        self.use_super_categories = False  # TODO
        self.data_is_zipped = self.config['INPUT_AS_ZIP']
        self.do_preproc = self.config['DO_PREPROC']
        self.class_counter = 1

        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']

        # Get classes to eval
        self.valid_classes = self.class_name_to_class_id.keys()  # FIXME
        self.class_list = self.class_name_to_class_id.keys()
        self.valid_class_numbers = list(self.class_name_to_class_id.values())  # FIXME

        # Get sequences to eval and check gt files exist
        if len(self.seq_list) < 1:
            raise TrackEvalException('No sequences are selected to be evaluated.')

        # Check gt files exist
        for seq in self.seq_list:
            if not self.data_is_zipped:
                curr_file = self.config["GT_LOC_FORMAT"].format(gt_folder=self.gt_fol, seq=seq)
                if not os.path.isfile(curr_file):
                    print('GT file not found ' + curr_file)
                    raise TrackEvalException('GT file not found for sequence: ' + seq)
        if self.data_is_zipped:
            curr_file = os.path.join(self.gt_fol, 'data.zip')
            if not os.path.isfile(curr_file):
                print('GT file not found ' + curr_file)
                raise TrackEvalException('GT file not found: ' + os.path.basename(curr_file))

        # Get trackers to eval
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']

        if self.config['TRACKER_DISPLAY_NAMES'] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config['TRACKERS_TO_EVAL'] is not None) and (
                len(self.config['TRACKER_DISPLAY_NAMES']) == len(self.tracker_list)):
            self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))
        else:
            raise TrackEvalException('List of tracker files and tracker display names do not match.')

        for tracker in self.tracker_list:
            if self.data_is_zipped:
                curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
                if not os.path.isfile(curr_file):
                    print('Tracker file not found: ' + curr_file)
                    raise TrackEvalException('Tracker file not found: ' + tracker + '/' + os.path.basename(curr_file))
            else:
                for seq in self.seq_list:
                    curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.json')
                    if not os.path.isfile(curr_file):
                        print('Tracker file not found: ' + curr_file)
                        raise TrackEvalException(
                            'Tracker file not found: ' + tracker + '/' + self.tracker_sub_fol + '/' + os.path.basename(
                                curr_file))

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _get_seq_info(self):
        if self.config["SEQ_INFO"]:
            seq_list = list(self.config["SEQ_INFO"].keys())
            seq_lengths = self.config["SEQ_INFO"]
        else:
            if self.config["SEQMAP_FILE"]:
                seqmap_file = self.config["SEQMAP_FILE"]
            else:
                if self.config["SEQMAP_FOLDER"] is None:
                    seqmap_file = os.path.join(self.config['GT_FOLDER'], self.gt_set + '.txt')
                else:
                    seqmap_file = os.path.join(self.config["SEQMAP_FOLDER"], self.gt_set + '.txt')

            if not os.path.isfile(seqmap_file):
                print('no seqmap found: ' + seqmap_file)
                raise TrackEvalException('no seqmap found: ' + os.path.basename(seqmap_file))
            with open(seqmap_file, 'r') as f:
                data = json.load(f)

            seq_list = [seq["name"] for seq in data]
            seq_lengths = {seq["name"]: seq["nframes"] for seq in data}
        return seq_list, seq_lengths

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the MOT Challenge 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        # File location
        if is_gt:
            file = self.config["GT_LOC_FORMAT"].format(gt_folder=self.gt_fol, seq=seq)
        else:
            file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.json')

        with open(file, 'r') as f:
            data = json.load(f)

        if is_gt:
            self.categories = {categ['id']: categ for categ in data["categories"]}
            self.images = data["images"]
            # Create a dictionary mapping from image_id to timestep
            self.image_id_to_timestep = {image["image_id"]: int(os.path.splitext(image["file_name"])[0]) - 1 for image in
                                    data["images"]}

        num_timesteps = len(self.images)  # FIXME what if unlabeled images?

        # Initialize lists with None for each timestep
        ids = [None] * num_timesteps
        classes = [None] * num_timesteps
        dets = [None] * num_timesteps
        crowd_ignore_regions = [None] * num_timesteps
        extras = [None] * num_timesteps
        confidences = [None] * num_timesteps

        # # detections = data["annotations"] if is_gt else data["predictions"]
        # # for annotation in detections:  # FIXME
        # for annotation in data["annotations"]:  # TODO remove
        #     if annotation["supercategory"] != "object":  # ignore pitch and camera
        #         continue
        #     role = annotation["attributes"]["role"]
        #     jersey_number = annotation["attributes"]["jersey"]
        #     team = annotation["attributes"]["team"]
        #     class_name = attributes_to_class_name(role, team, jersey_number)
        #     if class_name not in self.all_classes:
        #         self.all_classes[class_name] = {
        #             "id": self.class_counter,
        #             "name": class_name,
        #             "supercategory": "object"
        #         }
        #         self.class_counter += 1
        # list(self.all_classes.values())

        key = "annotations" if is_gt else "predictions"
        for annotation in data[key]:
            if annotation["supercategory"] != "object":  # ignore pitch and camera
                continue
            timestep = self.image_id_to_timestep[annotation["image_id"]]
            if ids[timestep] is None:
                ids[timestep] = []
                classes[timestep] = []
                dets[timestep] = []
                crowd_ignore_regions[timestep] = []
                extras[timestep] = []
                confidences[timestep] = []

            crowd_ignore_regions[timestep].append(np.empty((0, 4)))
            bbox_image = annotation["bbox_image"]  # FIXME use bbox_pitch and turn into bbox
            dets[timestep].append([bbox_image["x"], bbox_image["y"], bbox_image["w"], bbox_image["h"]])
            ids[timestep].append(annotation["track_id"])

            # confidence = annotation["confidence"] if not is_gt else 1
            confidence = 0.8 if not is_gt else 1  # FIXME
            confidences[timestep].append(confidence)

            # Extract extra information if needed (modify this part based on your requirements)
            role = annotation["attributes"]["role"]
            jersey_number = annotation["attributes"]["jersey"]
            team = annotation["attributes"]["team"]
            category_id = annotation["category_id"]
            extras[timestep].append({
                "role": role,
                "jersey": jersey_number,
                "team": team,
                "category_id": category_id,
                # Add more fields as needed
            })

            if self.eval_mode == 'classes':
                class_name = attributes_to_class_name(role, team, jersey_number)
                class_id = self.class_name_to_class_id[class_name]
                # class_id = self.class_name_to_class_id[class_name] if class_name in self.class_name_to_class_id else -1
                classes[timestep].append(class_id)
            else:
                classes[timestep].append(1)

            

        # Convert lists to numpy arrays
        for t in range(num_timesteps):
            if ids[t] is not None:
                ids[t] = np.array(ids[t])
                classes[t] = np.array(classes[t])
                dets[t] = np.array(dets[t])
                crowd_ignore_regions[t] = np.array(crowd_ignore_regions[t])
                confidences[t] = np.array(confidences[t])

        if is_gt:
            raw_data = {
                "gt_classes": [np.array(x) for x in classes],
                "gt_crowd_ignore_regions": crowd_ignore_regions,
                "gt_dets": dets,
                "gt_extras": extras,
                "gt_ids": [np.array(x) for x in ids],
                "seq": seq,
                "num_timesteps": num_timesteps,
            }
        else:
            raw_data = {
                "tracker_classes": [np.array(x) for x in classes],
                "tracker_dets": dets,
                "tracker_ids": [np.array(x) for x in ids],
                "tracker_extras": extras,
                "tracker_confidences": [np.array(x) for x in confidences],
            }
            # raw_data = add_noise_to_data(raw_data, len(self.images))
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        BDD100K:
            In BDD100K, the 4 preproc steps are as follow:
                1) There are eight classes (pedestrian, rider, car, bus, truck, train, motorcycle, bicycle)
                    which are evaluated separately.
                2) For BDD100K there is no removal of matched tracker dets.
                3) Crowd ignore regions are used to remove unmatched detections.
                4) No removal of gt dets.
        """
        cls_id = self.class_name_to_class_id[cls]

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):

            # Only extract relevant dets for this class for preproc and eval (cls)
            gt_class_mask = np.atleast_1d(raw_data['gt_classes'][t] == cls_id)
            gt_class_mask = gt_class_mask.astype(bool)
            gt_ids = raw_data['gt_ids'][t][gt_class_mask]
            gt_dets = raw_data['gt_dets'][t][gt_class_mask]

            tracker_class_mask = np.atleast_1d(raw_data['tracker_classes'][t] == cls_id)
            tracker_class_mask = tracker_class_mask.astype(bool)
            tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
            tracker_dets = raw_data['tracker_dets'][t][tracker_class_mask]
            similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]

            data['tracker_ids'][t] = tracker_ids
            data['tracker_dets'][t] = tracker_dets
            data['gt_ids'][t] = gt_ids
            data['gt_dets'][t] = gt_dets  # FIXME assert not 0 size
            data['similarity_scores'][t] = similarity_scores

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']

        # Ensure that ids are unique per timestep.
        self._check_unique_ids(data)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='xywh')
        return similarity_scores

    @_timing.time
    def get_raw_seq_data(self, tracker, seq):
        """ Loads raw data (tracker and ground-truth) for a single tracker on a single sequence.
        Raw data includes all of the information needed for both preprocessing and evaluation, for all classes.
        A later function (get_processed_seq_data) will perform such preprocessing and extract relevant information for
        the evaluation of each class.

        This returns a dict which contains the fields:
        [num_timesteps]: integer
        [gt_ids, tracker_ids, gt_classes, tracker_classes, tracker_confidences]:
                                                                list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, tracker_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [similarity_scores]: list (for each timestep) of 2D NDArrays.
        [gt_extras]: dict (for each extra) of lists (for each timestep) of 1D NDArrays (for each det).

        gt_extras contains dataset specific information used for preprocessing such as occlusion and truncation levels.

        Note that similarities are extracted as part of the dataset and not the metric, because almost all metrics are
        independent of the exact method of calculating the similarity. However datasets are not (e.g. segmentation
        masks vs 2D boxes vs 3D boxes).
        We calculate the similarity before preprocessing because often both preprocessing and evaluation require it and
        we don't wish to calculate this twice.
        We calculate similarity between all gt and tracker classes (not just each class individually) to allow for
        calculation of metrics such as class confusion matrices. Typically the impact of this on performance is low.

        SoccerNet game state specificity: similarity score is set to 0 if the attributes (jersey number, team, ...) of the tracker and the ground do not match.
        """
        # Load raw data.
        raw_gt_data = self._load_raw_file(tracker, seq, is_gt=True)
        raw_tracker_data = self._load_raw_file(tracker, seq, is_gt=False)
        raw_data = {**raw_tracker_data, **raw_gt_data}  # Merges dictionaries

        # Calculate similarities for each timestep.
        similarity_scores = []
        for t, (gt_dets_t, tracker_dets_t, gt_extras_t, tracker_extras_t) in enumerate(
                zip(raw_data['gt_dets'], raw_data['tracker_dets'], raw_data['gt_extras'], raw_data['tracker_extras'])):
            ious = self._calculate_similarities(gt_dets_t, tracker_dets_t)

            if self.eval_mode == 'distance':
                # Set similarity score to 0 if attributes do not match
                for i, (gt_extra, tracker_extra) in enumerate(zip(gt_extras_t, tracker_extras_t)):
                    # if gt_extra['role'] != tracker_extra['role'] or gt_extra['team'] != tracker_extra['team'] or gt_extra['jersey'] != tracker_extra['jersey']:
                    if attributes_to_class_name(gt_extra['role'], gt_extra['team'], gt_extra['jersey']) != attributes_to_class_name(tracker_extra['role'], tracker_extra['team'], tracker_extra['jersey']):
                        ious[i] = 0

            similarity_scores.append(ious)
            # assert not np.any((ious != 1) & (ious != 0))
        raw_data['similarity_scores'] = similarity_scores
        return raw_data


def attributes_to_class_name(role, team, jersey_number):
    # if role == "goalkeeper":
    if "goalkeeper" in role:
        role = "goalkeeper"  # some are tagged as "goalkeepersS"
        category = f"{role}_{team}_{jersey_number}" if jersey_number is not None else f"{role}_{team}"
    elif role == "player":
        category = f"{role}_{team}_{jersey_number}" if jersey_number is not None else f"{role}_{team}"
    elif role == "referee":
        category = f"{role}"
    elif role == "ball":
        category = f"{role}"
    elif role == "other":
        category = f"{role}"
    else:
        category = f"unknown_{role}"
    return category


def extract_all_classes(config, gt_fol, seq_list):
    all_classes = {}
    for seq in seq_list:
        # File location
        file = config["GT_LOC_FORMAT"].format(gt_folder=gt_fol, seq=seq)

        with open(file, 'r') as f:
            data = json.load(f)

        for annotation in data["annotations"]:
            if annotation["supercategory"] != "object":  # ignore pitch and camera
                continue
            role = annotation["attributes"]["role"]
            jersey_number = annotation["attributes"]["jersey"]
            team = annotation["attributes"]["team"]
            class_name = attributes_to_class_name(role, team, jersey_number)
            if class_name not in all_classes:
                all_classes[class_name] = {
                    "id": len(all_classes) + 1,
                    "name": class_name,
                    "supercategory": "object"
                }
    return all_classes


def add_noise_to_data(raw_data, num_timesteps, proba=0.2):
    all_classes = np.unique(np.concatenate(raw_data['tracker_classes']))

    for t in range(num_timesteps):
        # if raw_data['tracker_classes'][t] is not None:
        #     for i in range(len(raw_data['tracker_classes'][t])):
        #         if random.random() < proba:
        #             raw_data['tracker_classes'][t][i] = random.choice(all_classes)

        if raw_data['tracker_dets'][t] is not None:
            for i in range(len(raw_data['tracker_dets'][t])):
                shift_x = raw_data['tracker_dets'][t][i][2] * random.uniform(-0.1, 0.1)
                shift_y = raw_data['tracker_dets'][t][i][3] * random.uniform(-0.1, 0.1)
                raw_data['tracker_dets'][t][i][0] += shift_x
                raw_data['tracker_dets'][t][i][1] += shift_y

        if random.random() < proba and len(raw_data['tracker_ids'][t]) > 0:
            remove_index = random.randint(0, len(raw_data['tracker_ids'][t]) - 1)
            raw_data['tracker_ids'][t] = np.delete(raw_data['tracker_ids'][t], remove_index)
            raw_data['tracker_classes'][t] = np.delete(raw_data['tracker_classes'][t], remove_index)
            raw_data['tracker_dets'][t] = np.delete(raw_data['tracker_dets'][t], remove_index, axis=0)
            raw_data['tracker_confidences'][t] = np.delete(raw_data['tracker_confidences'][t], remove_index)
            raw_data['tracker_extras'][t].pop(remove_index)

    return raw_data
