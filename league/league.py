import os


class League:
    def __init__(self, create_agent, save_path):
        self.save_path = save_path
        self.next_lvl = 1
        self.frozen_idx = 0

        self.challenger = create_agent()
        self.frozen_agent = create_agent(train_on=False)        
        if not self.restore_agents():
            print("initializing challenger")
            self.challenger.save("challenger")

    def restore_agents(self):
        if not os.path.exists(self.save_path): return False
        print("restoring challenger")
        agent_list = os.listdir(self.save_path)
        agent_count = len(agent_list)
        print("agent_count: {}".format(agent_count))
        self.next_lvl = agent_count
        self.challenger.restore('challenger', log=False)
        return True

    def freeze_challenger(self):
        self.challenger.save(str(self.next_lvl))
        self.next_lvl += 1

    def promote(self):
        print("PROMOTED TO LEVEL {}! ^_^".format(int(self.next_lvl)))
        self.freeze_challenger()

    def get_frozen_agent(self, idx):
        if idx <= 0: return None
        if idx == self.frozen_idx: return self.frozen_agent
        self.frozen_agent.restore(str(idx), log=False)
        self.frozen_idx = idx
        #print("Frozen agent is restored from {}.".format(idx))
        return self.frozen_agent

    def get_agent_count(self):
        return self.next_lvl

    @staticmethod
    def calc_expected_result(elo1, elo2):
        r1 = 10 ** (elo1 / 400.0)
        r2 = 10 ** (elo2 / 400.0)
        return r1 / (r1 + r2)
