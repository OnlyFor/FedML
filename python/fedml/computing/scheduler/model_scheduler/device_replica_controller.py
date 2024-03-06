import logging
import copy
from .device_model_cache import FedMLModelCache


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
        self.devices_avail_gpus = self.get_devices_avail_gpus_frm_request_json()
        self.e_id = self.get_eid_frm_request_json()
        self.total_gpu_num = self.calc_total_gpu_num()
        self.gpu_per_replica = self.get_gpu_per_replica_frm_request_json()
        self.min_replica_num = self.get_min_replica_num_frm_request_json()
        self.max_replica_num = self.get_max_replica_num_frm_request_json()
        self.target_replica_num = self.init_id_replica_num()
        self.curr_replica_num = self.get_curr_replica_num_state_frm_db()
        self.intermediate_replica_num = copy.deepcopy(self.curr_replica_num)

        self.endpoint_name = self.get_endpoint_name_frm_request_json()
        self.model_name = self.get_model_name_frm_request_json()

    def get_devices_avail_gpus_frm_request_json(self):
        if "gpu_topology" not in self.request_json:
            # TODO: raise error, now using default value
            gpu_topology = {}
            for id in self.request_json["device_ids"]:
                if str(id) == str(self.master_id):
                    continue
                gpu_topology[id] = 2
            return gpu_topology
        return self.request_json["gpu_topology"]

    def get_eid_frm_request_json(self):
        if "end_point_id" not in self.request_json:
            raise ValueError("end_point_id is not in request_json")
        return self.request_json["end_point_id"]

    def get_endpoint_name_frm_request_json(self):
        if "end_point_name" not in self.request_json:
            raise ValueError("end_point_name is not in request_json")
        return self.request_json["end_point_name"]

    def get_model_name_frm_request_json(self):
        if "model_name" not in self.request_json["model_config"]:
            raise ValueError("model_name is not in request_json")
        return self.request_json["model_config"]["model_name"]

    def calc_total_gpu_num(self):
        total_gpu_num = 0
        for device_id, gpu_num in self.devices_avail_gpus.items():
            total_gpu_num += gpu_num
        return total_gpu_num

    def get_gpu_per_replica_frm_request_json(self):
        """
        Read gpu_per_replica from user's config yaml file. Default 1.
        """
        if "gpu_per_replica" in self.request_json["parameters"]:
            return self.request_json["parameters"]["gpu_per_replica"]
        return 1

    def get_min_replica_num_frm_request_json(self):
        """
        Read min_replica_num from mlops config yaml file. Default 0.
        """
        if "instance_scale_min" in self.request_json["model_config"]:
            return self.request_json["model_config"]["instance_scale_min"]
        return 0

    def get_max_replica_num_frm_request_json(self):
        """
        Read min_replica_num from mlops config yaml file. Default the number of gpu available.
        """
        if "instance_scale_max" in self.request_json["model_config"]:
            return self.request_json["model_config"]["instance_scale_max"]
        return self.total_gpu_num

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
        [
            # Replica 1
            {'end_point_id': , 'end_point_name': '', 'model_id': , 'model_name':
            '', 'model_url': '', 'model_version': ',
            'port': , 'inference_engine': , 'model_metadata': {}, 'model_config': None, 'model_status': '',
            'inference_port': , 'replica_no': , 'is_retain': },

            ...
        ]
        """
        # TODO: Change to deployment result to support the continuous deployment
        res_frm_db = FedMLModelCache.get_instance().get_endpoint_devices_replica_num(self.e_id)

        if res_frm_db is None:
            # First time to get the replica number from the database
            res_frm_db = {}
            for id, target_num in self.target_replica_num.items():
                res_frm_db[str(id)] = 0

        return res_frm_db

    def generate_diff_to_request_json(self):
        """
        Write the diff (curr <> target) to the self.request_json. e.g.
        {
            "replica_num_diff": {
                id1: {"op": "add", "curr_num": 1, "target_num": 2},
                id2: {"op": "add", "curr_num": 1, "target_num": 2}
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

    def callback_update_curr_replica_num_state(self, changed_device_id, replica_no):
        """
        Callback function to update the current replica number.
        curr_state: {id1: 1, id2: 1}
        target_replica_state = {id1: 2, id2: 2}
        intermediate_state = {id1: 2, id2: 1}
        """
        if str(changed_device_id) not in self.intermediate_replica_num:
            raise ValueError(f"changed_device_id {changed_device_id} is not in intermediate_replica_num")
        self.intermediate_replica_num[str(changed_device_id)] += 1

    def is_all_replica_ready(self):
        """
        Check if all the replicas are ready.
        """
        for id, replica_no in self.intermediate_replica_num.items():
            if replica_no < self.target_replica_num[id]:
                return False
        return True
