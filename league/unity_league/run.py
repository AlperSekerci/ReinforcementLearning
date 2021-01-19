import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

if __name__ == "__main__":
    from unity_league.trainer import Trainer
    from envs.unity.my_unity.volleyball.volleyball import create_ppo_agent as create_agent, VISUALIZE, REW_TYPE

    trainer = Trainer(base_port=16920,
                      game_path=r"K:\Game Development\AI\volleyball\Builds\{}{}Trainer\Volleyball.exe".format("Visual" if VISUALIZE else "", REW_TYPE),
                      create_agent=create_agent,
                      algo="ppo",
                      env_name="Volleyball",
                      grind_step=128,
                      ranked_reset=0,
                      default_reset=0,
                      elo_interval=100,
                      king_count=4,
                      use_chal_as_op=False,
                      visualize=VISUALIZE,
                      reward_mul=1 if REW_TYPE == 'Hier' else 1e-2,
                      use_profiler=False,
                      #profiler_name="bqn",
                      plot_brain=not VISUALIZE,
                      )
    trainer.train()
    # restoring challenger from a saved agent
    agent_name = "2"
    agent = create_agent(train_on=False)
    agent.restore(agent_name)
    agent.save('challenger')
    exit()
