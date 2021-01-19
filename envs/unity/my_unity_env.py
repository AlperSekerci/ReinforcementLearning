import socket
import numpy as np
import struct
import subprocess

class MyUnityEnv:
    def __init__(self, port, game_path, agent_params, reset_mode=0):
        self.agent_params = agent_params
        self.obs_bytes = 8 * np.prod(agent_params.state_shape)
        self.mask_bytes = 0 if not agent_params.use_act_mask else 2 * agent_params.total_act_size
        self.state_bytes = self.obs_bytes + self.mask_bytes
        self.feedback_bytes = 5
        self.decision_req_bytes = self.state_bytes + self.feedback_bytes

        self.act_bytes = 2 * agent_params.branch_count
        self.meta_bytes = 1
        self.step_bytes = self.act_bytes + self.meta_bytes
        self.first_send = True

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('localhost', port)
        #print('starting up on {} port {}'.format(*server_address))
        sock.bind(server_address)
        sock.listen(1)

        # Start the game.
        subprocess.Popen([game_path, '-rl_port', str(port)])

        #print('waiting for a connection')
        self.connection, client_address = sock.accept()
        #print('connection from', client_address)
        self.connection.send(np.array([reset_mode], dtype=np.uint8).tobytes())

        self.obs = None
        self.mask = None
        self.rew = 0
        self.done = False
        self.result = -2
        data = np.frombuffer(self.connection.recv(self.state_bytes), dtype=np.uint8)
        self.separate_data(data, has_feedback=False)

    def separate_data(self, data, has_feedback):
        start = 0
        end = self.obs_bytes
        self.obs = np.frombuffer(data[start:end], dtype=np.float32).reshape([2] + self.agent_params.state_shape)
        #print("")
        #print("obs: {}".format(self.obs))
        start = end
        if self.agent_params.use_act_mask:
            end += self.mask_bytes
            self.mask = np.frombuffer(data[start:end], dtype=np.bool).reshape((2, self.agent_params.total_act_size)).astype(np.float32)
            #print("mask: {}".format(self.mask))
            start = end
        if has_feedback:
            end += 4
            self.rew = float(struct.unpack('f', data[start:end])[0])
            self.result = int(data[-1]) - 2
            self.done = self.result != -2
            #print("rew: {}\ndone: {}\nresult: {}".format(self.rew, self.done, self.result))

    def step(self, acts, reset_mode=0):
        data = bytearray(memoryview(acts.flatten().astype(np.uint8)))
        if not self.first_send: data.append(np.uint8(reset_mode))
        self.first_send = False
        self.connection.send(data)
        #print("sent: {}".format(data))
        data = np.frombuffer(self.connection.recv(self.decision_req_bytes), dtype=np.uint8)
        self.separate_data(data, has_feedback=True)
