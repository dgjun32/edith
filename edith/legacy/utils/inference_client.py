import numpy as np
from edith.legacy.utils.service import ExternalRobotInferenceClient
import threading
ACTION_HORIZON = 16

class Gr00tRobotInferenceAsyncClient:
    def __init__(
        self,
        host="localhost",
        port=5555,
        min_action_len=12,
        language_instruction="Pick up the fruits and place them on the plate.",
    ):
        self.min_action_len = min_action_len
        self.language_instruction = language_instruction
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

        self.action_index = 0
        self.obs_sent_index = 0
        self.action_chunk = None
        self.waiting_for_action_chunk = False

    def get_action(self, obs_dict):
        # 쓰레드로 비동기 액션 청크 요청
        def async_get_action_chunk(obs_dict, client):
            client.action_chunk = client.get_action_chunk(obs_dict)
            client.obs_sent_index = client.action_index
            client.waiting_for_action_chunk = False

        if not self.waiting_for_action_chunk and (self.action_chunk is None or self.action_index - self.obs_sent_index >= self.min_action_len):
            self.waiting_for_action_chunk = True
            thread = threading.Thread(target=async_get_action_chunk, args=(obs_dict, self))
            thread.start()

        if (self.action_chunk is None or 
            self.action_index - self.obs_sent_index >= ACTION_HORIZON):
            while self.waiting_for_action_chunk:
                pass  # busy wait until action chunk is ready
            
        action_dict = {}
        for key in self.action_chunk.keys():
            action = self.action_chunk[key][self.action_index - self.obs_sent_index]
            # 1D 이상으로 만듦 (최소 1D 보장)
            action_dict[key.replace("action.", "")] = np.atleast_1d(action)
        self.action_index += 1
        print(f"action_index: {self.action_index}, {self.action_index - self.obs_sent_index} / {ACTION_HORIZON}")
        return action_dict

    def get_action_chunk(self, obs_dict):
        res = self.policy.get_action(obs_dict)
        # print("Inference query time taken", time.time() - start_time)
        return res

    def set_lang_instruction(self, lang_instruction):
        self.language_instruction = lang_instruction

class Gr00tRobotInferenceSyncClient:
    def __init__(
        self,
        host="localhost",
        port=5555,
        action_horizon=ACTION_HORIZON,
        language_instruction="Pick up the fruits and place them on the plate.",
    ):
        self.action_horizon = action_horizon
        self.language_instruction = language_instruction
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.subgoal_id = 0

        self.action_index = 0
        self.action_chunk = None

    def get_action(self, obs_dict):
        if self.action_index == 0:
            self.action_chunk = self.get_action_chunk(obs_dict)
        
        action_dict = {}
        for key in self.action_chunk.keys():
            action = self.action_chunk[key][self.action_index]
            # 1D 이상으로 만듦 (최소 1D 보장)
            action_dict[key.replace("action.", "")] = np.atleast_1d(action)
        self.action_index += 1
        print(f"action_index: {self.action_index}")
        if self.action_index == self.action_horizon:
            self.action_index = 0
        return action_dict

    def get_action_chunk(self, obs_dict):
        res = self.policy.get_action(obs_dict)
        # print("Inference query time taken", time.time() - start_time)
        return res

    
    def get_action_with_subgoal_id(self, obs_dict):
        if self.action_index == 0:
            self.action_chunk = self.get_action_chunk_with_subgoal_id(obs_dict)
        
        action_dict = {}
        for key in self.action_chunk.keys():
            action = self.action_chunk[key][self.action_index]
            # 1D 이상으로 만듦 (최소 1D 보장)
            action_dict[key.replace("action.", "")] = np.atleast_1d(action)
        self.action_index += 1
        print(f"action_index: {self.action_index}")
        if self.action_index == self.action_horizon:
            self.action_index = 0
        return action_dict


    def get_action_chunk_with_subgoal_id(self, obs_dict):
        res = self.policy.get_action_with_subgoal_id(obs_dict, self.subgoal_id)
        # print("Inference query time taken", time.time() - start_time)
        return res

    def set_lang_instruction(self, lang_instruction):
        self.language_instruction = lang_instruction



class Gr00tRobotInferencewithGenSyncClient:
    def __init__(
        self,
        host="localhost",
        port=5555,
        action_horizon=ACTION_HORIZON,
        language_instruction="Pick up the fruits and place them on the plate.",
    ):
        self.action_horizon = action_horizon
        self.language_instruction = language_instruction
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

        self.action_index = 0
        self.action_chunk = None

    def get_action(self, obs_dict):
        if self.action_index == 0:
            self.action_chunk, self.generated_tokens = self.get_action_chunk(obs_dict)
        
        action_dict = {}
        for key in self.action_chunk.keys():
            action = self.action_chunk[key][self.action_index]
            # 1D 이상으로 만듦 (최소 1D 보장)
            action_dict[key.replace("action.", "")] = np.atleast_1d(action)
        self.action_index += 1
        print(f"action_index: {self.action_index}")
        if self.action_index == self.action_horizon:
            self.action_index = 0
        return action_dict, self.generated_tokens

    def get_action_chunk(self, obs_dict):
        action_chunk, generated_tokens = self.policy.get_action(obs_dict)
        # print("Inference query time taken", time.time() - start_time)
        return action_chunk, generated_tokens

    def set_lang_instruction(self, lang_instruction):
        self.language_instruction = lang_instruction



class PiRobotClient:
    def __init__(
        self,
        host="localhost",
        port=5555,
        action_horizon=ACTION_HORIZON,
        language_instruction="Pick up the fruits and place them on the plate.",
    ):
        self.action_horizon = action_horizon
        self.language_instruction = language_instruction
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.action_index = 0
        self.action_chunk = None

    def get_action(self, obs_dict):
        if self.action_index == 0:
            self.action_chunk = self.get_action_chunk(obs_dict)
        
        action_dict = {}
        for key in self.action_chunk.keys():
            action = self.action_chunk[key][self.action_index]
            # 1D 이상으로 만듦 (최소 1D 보장)
            action_dict[key.replace("action.", "")] = np.atleast_1d(action)
        self.action_index += 1
        print(f"action_index: {self.action_index}")
        if self.action_index == self.action_horizon:
            self.action_index = 0
        return action_dict

    def get_action_chunk(self, obs_dict):
        res = self.policy.get_action(obs_dict)
        # print("Inference query time taken", time.time() - start_time)
        return res

    def set_lang_instruction(self, lang_instruction):
        self.language_instruction = lang_instruction