import logging
from ..scheduler_core.compute_cache_manager import ComputeCacheManager


class FedMLDeviceReplicaHandler:
    def __init__(self, worker_id, request_json: dict):
        """
        Handler on the worker to acutally exec the reconcilation logic (Incluing add, remove, update).

        e_id: unique id (i.e. endpoint_id) for each deployment
        devices_avail_gpus = {device_id1: gpu_num, device_id2: gpu_num, ...}
        request_json: json from MLOps for this deployment
        total_gpu_num: total number of gpus will be used for this deployment
        gpu_per_replica: number of gpus required per replica
        """
        self.worker_id = worker_id
        self.request_json = request_json
        self.e_id = self.get_eid_frm_request_json()
        self.gpu_per_replica = self.get_gpu_per_replica_frm_request_json()
        self.replica_num_diff = self.get_diff_replica_num_frm_request_json()

        self.device_avail_gpus = self.get_device_avail_gpus_frm_db()

    def get_device_avail_gpus_frm_db(self):
        """
        Get the available gpus from db.
        """
        available_gpu_ids = ComputeCacheManager.get_instance().get_gpu_cache().get_device_available_gpu_ids(
            self.worker_id)
        logging.info(f"[Replica Handler] [endpoint {self.e_id} ] [worker {self.worker_id}] "
                     f"All device_avail_gpus: {available_gpu_ids}")
        return available_gpu_ids

    def get_eid_frm_request_json(self):
        if "end_point_id" not in self.request_json:
            raise ValueError("end_point_id is not in request_json")
        return self.request_json["end_point_id"]

    def get_gpu_per_replica_frm_request_json(self):
        """
        Read gpu_per_replica from user's config yaml file. Default 1.
        """
        if "gpu_per_replica" in self.request_json["parameters"]:
            return self.request_json["parameters"]["gpu_per_replica"]
        return 1

    def get_diff_replica_num_frm_request_json(self):
        """
        Read replica_diff passing by master's request json.
        Return:
        {
            id1_str: {"op": "add", "curr_num": 1, "target_num": 2},
            id2_str: {"op": "add", "curr_num": 1, "target_num": 2}
        }
        """
        if "replica_num_diff" in self.request_json and str(self.worker_id) in self.request_json["replica_num_diff"]:
            return self.request_json["replica_num_diff"][str(self.worker_id)]
        return None

    def reconcile_num_replica(self):
        """
        To solve the conflict between different reconciliation requests. The request & delete reqs should be
        executed in order and atomic (i.e. rollback).

        return (op, number of op)
        """
        if not self.replica_num_diff:
            raise ValueError(f"replica_num_diff is empty, cannot reconcile.")

        if self.replica_num_diff["op"] not in ["add", "remove"]:
            raise ValueError(f"op should be add or remove. Got {self.replica_num_diff['op']}")

        if self.replica_num_diff["op"] == "add":
            op, op_num = (self.replica_num_diff["op"],
                          self.replica_num_diff["target_num"] - self.replica_num_diff["curr_num"])
        else:
            op, op_num = (self.replica_num_diff["op"],
                          self.replica_num_diff["curr_num"] - self.replica_num_diff["target_num"])
        return op, op_num
