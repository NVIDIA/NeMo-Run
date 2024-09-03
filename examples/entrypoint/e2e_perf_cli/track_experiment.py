import nemo_run as run

experiment = run.Experiment.from_id("pretrain-llama3-8b_1725263510")

print(experiment.tasks[0].status(experiment._runner), experiment.app_id)
# experiment.status() # Gets the overall status
# experiment.logs("nemo.collections.llm.api.train") # Gets the log for the provided task
