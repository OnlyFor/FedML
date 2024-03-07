import logging
import copy
from .device_model_cache import FedMLModelCache
from .device_model_msg_object import FedMLModelMsgObject
from .device_client_constants import ClientConstants


class FedMLDeviceReplicaController:
    def __init__(self, master_id, request_json: dict):
        """
        For each deployment, we have:
        master_id: unique id for the master device
        e_id: unique id (i.e. endpoint_id) for each deployment
        devices_avail_gpus = {device_id1: gpu_num, device_id2: gpu_num, ...}
        request_json: json from MLOps for this deployment
        total_gpu_num: total number of gpus will be used for this deployment
        gpu_per_replica: number of gpus required per replica
        min_replica_num: minimum number of replicas required
        max_replica_num: maximum number of replicas required
        """
        self.master_id = master_id
        self.request_json = request_json
        self.request_msg_obj = FedMLModelMsgObject("replica_controller", request_json)

        self.e_id = self.request_msg_obj.run_id
        self.devices_avail_gpus = self.request_msg_obj.gpu_topology
        self.total_gpu_num = self.calc_total_gpu_num()
        self.gpu_per_replica = self.request_msg_obj.gpu_per_replica
        self.min_replica_num = self.request_msg_obj.scale_min
        self.max_replica_num = self.request_msg_obj.scale_max
        self.endpoint_name = self.request_msg_obj.end_point_name
        self.model_name = self.request_msg_obj.model_name

        self.target_replica_num = self.init_id_replica_num()
        self.curr_replica_num = self.get_curr_replica_num_state_frm_db()
        self.intermediate_replica_num = copy.deepcopy(self.curr_replica_num)

    def calc_total_gpu_num(self):
        total_gpu_num = 0
        for device_id, gpu_num in self.devices_avail_gpus.items():
            total_gpu_num += gpu_num
        return total_gpu_num

    def init_id_replica_num(self):
        """
        Initialize the target replica number for each device.
        id_replica_num[id] = avail_num // self.gpu_per_replica
        """
        id_replica_num = {}
        for id, avail_num in self.devices_avail_gpus.items():
            if avail_num % self.gpu_per_replica != 0:
                raise ValueError("The number of gpus for each device should be divisible by gpu_per_replica")
            id_replica_num[str(id)] = avail_num // self.gpu_per_replica
        return id_replica_num

    def diff_target_curr_replica_num(self):
        logging.info(f"[Replica Controller] [endpoint {self.e_id} ]target_replica_state: {self.target_replica_num}")
        logging.info(f"[Replica Controller] [endpoint {self.e_id} ]curr_replica_state: {self.curr_replica_num}")
        diff = self.diff_target_curr_replica_num_impl(self.target_replica_num, self.curr_replica_num)
        logging.info(
            f"[Replica Controller] [endpoint {self.e_id} ]diff_target_curr_replica_num: {diff}")
        return diff

    @staticmethod
    def diff_target_curr_replica_num_impl(target_replica_state, curr_replica_state):
        """
        Return the difference between target and current replica number.
        "op" could only be "add" or "remove".
        e.g.
        curr_replica_state = {id1: 1, id2: 1}
        target_replica_state = {id1: 2, id2: 2}

        return {id1: {"op": "add", "curr_num": 1, "target_num": 2}, id2: {"op": "add", "curr_num": 1, "target_num": 2}}
        """
        diff_target_curr_replica_num = {}
        assert target_replica_state is not None

        if curr_replica_state is None:
            curr_replica_state = {}
            for id, target_num in target_replica_state.items():
                diff_target_curr_replica_num[id] = {"op": "add", "curr_num": 0, "target_num": target_num}
            return diff_target_curr_replica_num

        for id, target_num in target_replica_state.items():
            if target_num > curr_replica_state[id]:
                diff_target_curr_replica_num[id] = {"op": "add", "curr_num": curr_replica_state[id],
                                                    "target_num": target_num}
            elif target_num < curr_replica_state[id]:
                diff_target_curr_replica_num[id] = {"op": "remove", "curr_num": curr_replica_state[id],
                                                    "target_num": target_num}
            else:
                pass

        for id, curr_num in curr_replica_state.items():
            if id not in target_replica_state:
                diff_target_curr_replica_num[id] = {"op": "remove", "num": curr_num}

        return diff_target_curr_replica_num

    def diff_target_curr_replica_version(self, target_replica_version, curr_replica_version):
        """
        Return the difference between target and current replica version.
        "op" could only be "update".
        e.g.
        curr_replica_version = {id1: "v1", id2: "v1"}
        target_replica_version = {id1: "v2", id2: "v2"}

        return {id1: {"op": "update", "new_version": "v2", "old_version": "v1"},
        id2: {"op": "update", "new_version": "v2", "old_version": "v1"}}
        """
        # TODO: Finished the rolling update logic here
        pass

    def get_curr_replica_num_state_frm_db(self):
        """
        Sync the current replica number state from the database.
        Return the current replica number state.
        """
        res_frm_db = FedMLModelCache.get_instance().get_deployment_result_list(
            self.e_id, self.endpoint_name, self.model_name)

        curr_state = {}
        if res_frm_db is None or len(res_frm_db) == 0:
            # First time to get the replica number from the database
            for id, target_num in self.target_replica_num.items():
                curr_state[str(id)] = 0
        else:
            for result_item in res_frm_db:
                # Unpack the result_item
                result_device_id, _, result_payload = FedMLModelCache.get_instance().get_result_item_info(result_item)
                curr_state[str(result_device_id)] = curr_state.get(str(result_device_id), 0) + 1
        return curr_state

    def generate_diff_to_request_json(self):
        """
        Write the diff (curr <> target) to the self.request_json. e.g.
        {
            "replica_num_diff": {
                id1: {"op": "add", "curr_num": 1, "target_num": 2},
                id2: {"op": "add", "curr_num": 1, "target_num": 2},
                id3: {"op": "remove", "curr_num": 1, "target_num": 0}
            },
            "gpus_per_replica": 1,
        }
        """
        replica_diff_key = "replica_num_diff"
        gpu_per_replica_key = "gpus_per_replica"
        replica_diff = self.diff_target_curr_replica_num()
        self.request_json[replica_diff_key] = replica_diff
        self.request_json[gpu_per_replica_key] = self.gpu_per_replica
        return self.request_json

    def callback_update_curr_replica_num_state(self, changed_device_id, replica_no, op_type):
        """
        Callback function to update the current replica number.
        curr_state: {id1: 1, id2: 1}
        target_replica_state = {id1: 2, id2: 2}
        intermediate_state = {id1: 2, id2: 1}
        """
        if str(changed_device_id) not in self.intermediate_replica_num:
            raise ValueError(f"changed_device_id {changed_device_id} is not in intermediate_replica_num")

        if op_type == ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED:
            self.intermediate_replica_num[str(changed_device_id)] += 1
        elif op_type == ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DELETED:
            self.intermediate_replica_num[str(changed_device_id)] -= 1

    def is_all_replica_target_state(self):
        """
        Check if all the replicas are ready. Including the number and version.
        """
        for id, replica_no in self.intermediate_replica_num.items():
            if replica_no != self.target_replica_num[id]:
                return False
        return True
