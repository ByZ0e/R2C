import copy
import numpy as np

from collections import Counter, OrderedDict
from ai2thor.controller import Controller

from alfred import constants
import alfred.utils.game_util as game_util
from alfred.env.tasks import get_task

DEFAULT_RENDER_SETTINGS = {'renderImage': True,
                           'renderDepthImage': False,
                           'renderClassImage': False,
                           'renderObjectImage': False}

class ThorEnv(Controller):
    '''
    an extension of ai2thor.controller.Controller for ALFRED tasks
    '''
    def __init__(self,
                 x_display,
                 player_screen_height=constants.DETECTION_SCREEN_HEIGHT,
                 player_screen_width=constants.DETECTION_SCREEN_WIDTH,
                 quality='MediumCloseFitShadows',
                 build_path=constants.BUILD_PATH):
        super().__init__(quality=quality)
        self.local_executable_path = build_path
        self.start(x_display=str(x_display),
                   player_screen_height=player_screen_height,
                   player_screen_width=player_screen_width)
        self.task = None

        # internal states
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

        print("ThorEnv started.")

    def reset(self, scene_name_or_num,
              grid_size=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
              camera_y=constants.CAMERA_HEIGHT_OFFSET,
              render_image=constants.RENDER_IMAGE,
              render_depth_image=constants.RENDER_DEPTH_IMAGE,
              render_class_image=constants.RENDER_CLASS_IMAGE,
              render_object_image=constants.RENDER_OBJECT_IMAGE,
              visibility_distance=constants.VISIBILITY_DISTANCE,
              silent=False):
        '''
        reset scene and task states
        '''
        if not silent:
            print("Resetting ThorEnv")

        if type(scene_name_or_num) == str:
            scene_name = scene_name_or_num
        else:
            scene_name = 'FloorPlan%d' % scene_name_or_num

        super().reset(scene_name)
        event = super().step(dict(
            action='Initialize',
            gridSize=grid_size,
            cameraY=camera_y,
            renderImage=render_image,
            renderDepthImage=render_depth_image,
            renderClassImage=render_class_image,
            renderObjectImage=render_object_image,
            visibility_distance=visibility_distance,
            makeAgentsVisible=False,
        ))

        # reset task if specified
        if self.task is not None:
            self.task.reset()
        # clear object state changes
        self.reset_states()
        self.last_interaction = (None, None)

        return event

    def reset_states(self):
        '''
        clear state changes
        '''
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

    def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
        '''
        restore object locations and states
        '''
        super().step(dict(
            action='Initialize',
            gridSize=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
            cameraY=constants.CAMERA_HEIGHT_OFFSET,
            renderImage=constants.RENDER_IMAGE,
            renderDepthImage=constants.RENDER_DEPTH_IMAGE,
            renderClassImage=constants.RENDER_CLASS_IMAGE,
            renderObjectImage=constants.RENDER_OBJECT_IMAGE,
            visibility_distance=constants.VISIBILITY_DISTANCE,
            makeAgentsVisible=False,
        ))
        if len(object_toggles) > 0:
            super().step((dict(action='SetObjectToggles', objectToggles=object_toggles)))

        if dirty_and_empty:
            super().step(dict(action='SetStateOfAllObjects',
                               StateChange="CanBeDirty",
                               forceAction=True))
            super().step(dict(action='SetStateOfAllObjects',
                               StateChange="CanBeFilled",
                               forceAction=False))
        super().step((dict(action='SetObjectPoses', objectPoses=object_poses)))

    def set_task(self, traj, reward_type='sparse', max_episode_length=2000):
        '''
        set the current task type (one of 7 tasks)
        '''
        task_type = traj['task_type']
        self.task = get_task(
            task_type, traj, self, reward_type=reward_type,
            max_episode_length=max_episode_length)

    def step(self, action, smooth_nav=False):
        '''
        overrides ai2thor.controller.Controller.step() for smooth navigation and goal_condition updates
        '''
        if smooth_nav:
            if "MoveAhead" in action['action']:
                self.smooth_move_ahead(action)
            elif "Rotate" in action['action']:
                self.smooth_rotate(action)
            elif "Look" in action['action']:
                self.smooth_look(action)
            else:
                super().step(action)
        else:
            if "LookUp" in action['action']:
                self.look_angle(-constants.AGENT_HORIZON_ADJ)
            elif "LookDown" in action['action']:
                self.look_angle(constants.AGENT_HORIZON_ADJ)
            else:
                super().step(action)

        event = self.update_states(action)
        self.check_post_conditions(action)
        return event

    def check_post_conditions(self, action):
        '''
        handle special action post-conditions
        '''
        if action['action'] == 'ToggleObjectOn':
            self.check_clean(action['objectId'])

    def update_states(self, action):
        '''
        extra updates to metadata after step
        '''
        # add 'cleaned' to all object that were washed in the sink
        event = self.last_event
        if event.metadata['lastActionSuccess']:
            # clean
            if action['action'] == 'ToggleObjectOn' and "Faucet" in action['objectId']:
                sink_basin = game_util.get_obj_of_type_closest_to_obj(
                    'SinkBasin', action['objectId'], event.metadata)
                cleaned_object_ids = sink_basin['receptacleObjectIds']
                self.cleaned_objects = self.cleaned_objects | set(cleaned_object_ids) if cleaned_object_ids is not None else set()
            # heat
            if action['action'] == 'ToggleObjectOn' and "Microwave" in action['objectId']:
                microwave = game_util.get_objects_of_type(
                    'Microwave', event.metadata)[0]
                heated_object_ids = microwave['receptacleObjectIds']
                self.heated_objects = self.heated_objects | set(heated_object_ids) if heated_object_ids is not None else set()
            # cool
            if action['action'] == 'CloseObject' and "Fridge" in action['objectId']:
                fridge = game_util.get_objects_of_type('Fridge', event.metadata)[0]
                cooled_object_ids = fridge['receptacleObjectIds']
                self.cooled_objects = self.cooled_objects | set(cooled_object_ids) if cooled_object_ids is not None else set()

        return event

    def get_transition_reward(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for transition_reward")
        else:
            return self.task.transition_reward(self.last_event)

    def get_goal_satisfied(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return bool(self.task.goal_satisfied(self.last_event))

    def get_goal_conditions_met(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_conditions_met(self.last_event)

    def get_subgoal_idx(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for subgoal_idx")
        else:
            return self.task.get_subgoal_idx()

    def noop(self):
        '''
        do nothing
        '''
        super().step(dict(action='Pass'))

    def smooth_move_ahead(self, action, render_settings=None):
        '''
        smoother MoveAhead
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        smoothing_factor = constants.RECORD_SMOOTHING_FACTOR
        new_action = copy.deepcopy(action)
        new_action['moveMagnitude'] = constants.AGENT_STEP_SIZE / smoothing_factor

        new_action['renderImage'] = render_settings['renderImage']
        new_action['renderClassImage'] = render_settings['renderClassImage']
        new_action['renderObjectImage'] = render_settings['renderObjectImage']
        new_action['renderDepthImage'] = render_settings['renderDepthImage']

        events = []
        for xx in range(smoothing_factor - 1):
            event = super().step(new_action)
            if event.metadata['lastActionSuccess']:
                events.append(event)
        event = super().step(new_action)
        if event.metadata['lastActionSuccess']:
            events.append(event)
        return events

    def smooth_rotate(self, action, render_settings=None):
        '''
        smoother RotateLeft and RotateRight
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        if action['action'] == 'RotateLeft':
            end_rotation = (start_rotation - 90)
        else:
            end_rotation = (start_rotation + 90)

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': horizon,
                    'tempRenderChange': True,
                    'renderNormalsImage': False,
                    'renderImage': render_settings['renderImage'],
                    'renderClassImage': render_settings['renderClassImage'],
                    'renderObjectImage': render_settings['renderObjectImage'],
                    'renderDepthImage': render_settings['renderDepthImage'],
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': horizon,
                }
                event = super().step(teleport_action)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def smooth_look(self, action, render_settings=None):
        '''
        smoother LookUp and LookDown
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + constants.AGENT_HORIZON_ADJ * (1 - 2 * int(action['action'] == 'LookUp'))
        position = event.metadata['agent']['position']

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': rotation,
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                    'tempRenderChange': True,
                    'renderNormalsImage': False,
                    'renderImage': render_settings['renderImage'],
                    'renderClassImage': render_settings['renderClassImage'],
                    'renderObjectImage': render_settings['renderObjectImage'],
                    'renderDepthImage': render_settings['renderDepthImage'],
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': rotation,
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                }
                event = super().step(teleport_action)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def look_angle(self, angle, render_settings=None):
        '''
        look at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + angle
        position = event.metadata['agent']['position']

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': rotation,
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': np.round(end_horizon, 3),
            'tempRenderChange': True,
            'renderNormalsImage': False,
            'renderImage': render_settings['renderImage'],
            'renderClassImage': render_settings['renderClassImage'],
            'renderObjectImage': render_settings['renderObjectImage'],
            'renderDepthImage': render_settings['renderDepthImage'],
        }
        event = super().step(teleport_action)
        return event

    def rotate_angle(self, angle, render_settings=None):
        '''
        rotate at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        end_rotation = start_rotation + angle

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': np.round(end_rotation, 3),
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': horizon,
            'tempRenderChange': True,
            'renderNormalsImage': False,
            'renderImage': render_settings['renderImage'],
            'renderClassImage': render_settings['renderClassImage'],
            'renderObjectImage': render_settings['renderObjectImage'],
            'renderDepthImage': render_settings['renderDepthImage'],
        }
        event = super().step(teleport_action)
        return event

    def to_thor_api_exec(self, action, object_id="", smooth_nav=False):
        # TODA: parametrized navigation commands

        if "RotateLeft" in action:
            action = dict(action="RotateLeft",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "RotateRight" in action:
            action = dict(action="RotateRight",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "MoveAhead" in action:
            action = dict(action="MoveAhead",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "LookUp" in action:
            action = dict(action="LookUp",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "LookDown" in action:
            action = dict(action="LookDown",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "OpenObject" in action:
            action = dict(action="OpenObject",
                          objectId=object_id,
                          moveMagnitude=1.0)
            event = self.step(action)
        elif "CloseObject" in action:
            action = dict(action="CloseObject",
                          objectId=object_id,
                          forceAction=True)
            event = self.step(action)
        elif "PickupObject" in action:
            action = dict(action="PickupObject",
                          objectId=object_id)
            event = self.step(action)
        elif "PutObject" in action:
            inventory_object_id = self.last_event.metadata['inventoryObjects'][0]['objectId']
            action = dict(action="PutObject",
                          objectId=inventory_object_id,
                          receptacleObjectId=object_id,
                          forceAction=True,
                          placeStationary=True)
            event = self.step(action)
        elif "ToggleObjectOn" in action:
            action = dict(action="ToggleObjectOn",
                          objectId=object_id)
            event = self.step(action)

        elif "ToggleObjectOff" in action:
            action = dict(action="ToggleObjectOff",
                          objectId=object_id)
            event = self.step(action)
        elif "SliceObject" in action:
            # check if agent is holding knife in hand
            inventory_objects = self.last_event.metadata['inventoryObjects']
            if len(inventory_objects) == 0 or 'Knife' not in inventory_objects[0]['objectType']:
                raise Exception("Agent should be holding a knife before slicing.")

            action = dict(action="SliceObject",
                          objectId=object_id)
            event = self.step(action)
        else:
            raise Exception("Invalid action. Conversion to THOR API failed! (action='" + str(action) + "')")

        return event, action

    def check_clean(self, object_id):
        '''
        Handle special case when Faucet is toggled on.
        In this case, we need to execute a `CleanAction` in the simulator on every object in the corresponding
        basin. This is to clean everything in the sink rather than just things touching the stream.
        '''
        event = self.last_event
        if event.metadata['lastActionSuccess'] and 'Faucet' in object_id:
            # Need to delay one frame to let `isDirty` update on stream-affected.
            event = self.step({'action': 'Pass'})
            sink_basin_obj = game_util.get_obj_of_type_closest_to_obj(
                'SinkBasin', object_id, event.metadata)
            for in_sink_obj_id in sink_basin_obj['receptacleObjectIds']:
                if (game_util.get_object(in_sink_obj_id, event.metadata)['dirtyable']
                        and game_util.get_object(in_sink_obj_id, event.metadata)['isDirty']):
                    event = self.step({'action': 'CleanObject', 'objectId': in_sink_obj_id})
        return event

    @staticmethod
    def prune_by_any_interaction(instances_ids, all_objects):
        '''
        ignores any object that is not interactable in anyway
        '''
        pruned_instance_ids = []
        for obj in all_objects:
            obj_id = obj['objectId']
            if obj_id in instances_ids:
                if obj['pickupable'] or obj['receptacle'] or obj['openable'] or obj['toggleable'] or obj['sliceable']:
                    pruned_instance_ids.append(obj_id)

        ordered_instance_ids = [id for id in instances_ids if id in pruned_instance_ids]
        return ordered_instance_ids

    @staticmethod
    def mask_to_object(mask, last_event, debug=False, mask_px_sample=1):
        '''
        retreive object index from the mask interaction and segmetnation frame
        '''
        # ground-truth instance segmentation mask from THOR
        instance_segs = np.array(last_event.instance_segmentation_frame)
        color_to_object_id = last_event.color_to_object_id

        # get object_id for each 1-pixel in the interact_mask
        nz_rows, nz_cols = np.nonzero(mask)
        instance_counter = Counter()
        for i in range(0, len(nz_rows), mask_px_sample):
            x, y = nz_rows[i], nz_cols[i]
            instance = tuple(instance_segs[x, y])
            instance_counter[instance] += 1

        # iou scores for all instances
        iou_scores = {}
        for color_id, intersection_count in instance_counter.most_common():
            union_count = np.sum(
                np.logical_or(np.all(instance_segs == color_id, axis=2),
                              mask.astype(bool)))
            iou_scores[color_id] = intersection_count / float(union_count)
        iou_sorted_instance_ids = list(
            OrderedDict(sorted(iou_scores.items(), key=lambda x: x[1], reverse=True)))

        # get the most common object ids ignoring the object-in-hand
        inv_obj = last_event.metadata['inventoryObjects'][0]['objectId'] \
            if len(last_event.metadata['inventoryObjects']) > 0 else None
        all_ids = [color_to_object_id[color_id] for color_id in iou_sorted_instance_ids
                   if color_id in color_to_object_id
                   and color_to_object_id[color_id] != inv_obj]
        instance_ids = [inst_id for inst_id in all_ids if inst_id is not None]
        # prune invalid instances like floors, walls, etc.
        instance_ids = ThorEnv.prune_by_any_interaction(
            instance_ids, last_event.metadata['objects'])

        # cv2 imshows to show image, segmentation mask, interact mask
        if debug:
            print("action_box", "instance_ids", instance_ids)

        if len(instance_ids) == 0:
            return None
        object_id = instance_ids[0]
        # the pretrained MaskRCNN checkpoint identifies both Sink/Bathtub and their
        # basin as the same class due to the training data preprocessing.
        # We correct it manually here.
        if object_id.startswith('Sink|') or object_id.startswith('Bathtub|'):
            basin_id = object_id + '|{}Basin'.format(object_id.split('|')[0])
            if basin_id in instance_ids:
                object_id = basin_id
        return object_id

    def va_interact(
            self, action, interact_mask=None, smooth_nav=True, debug=False):
        '''
        interact mask based action call
        '''
        target_instance_id = ''
        navig_action = (interact_mask is None)

        # object selection module
        if not navig_action:
            assert isinstance(interact_mask, (np.ndarray, np.int64))
            target_instance_id = ThorEnv.mask_to_object(
                interact_mask, self.last_event, debug)

        if not navig_action and target_instance_id is None:
            err = "Bad interact mask. Couldn't locate target object"
            success = False
            return success, None, None, err, None

        if debug:
            print("taking action {} on id {}".format(action, target_instance_id))
        try:
            event, api_action = self.to_thor_api_exec(
                action, target_instance_id, smooth_nav)
        except Exception as err:
            success = False
            return success, None, None, err, None

        if not event.metadata['lastActionSuccess']:
            success = False
            return (success, event, target_instance_id,
                    event.metadata['errorMessage'], api_action)

        success = True
        return success, event, target_instance_id, '', api_action
