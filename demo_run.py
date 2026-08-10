import OpenPromptInjection as PI
from OpenPromptInjection.utils import open_config

DATA_NUM = 5

# Target task: sentiment analysis on SST-2
target_task = PI.create_task(open_config('./configs/task_configs/sst2_config.json'), DATA_NUM)

# Local, no-API-key model: small Flan-T5 (CPU)
model_config = open_config('./configs/model_configs/flan_small_config.json')
model = PI.create_model(config=model_config)
model.print_model_info()

# Injected task: spam detection on SMS Spam Collection
inject_task = PI.create_task(open_config('./configs/task_configs/sms_spam_config.json'), DATA_NUM, for_injection=True)
attacker = PI.create_attacker('combine', inject_task)

# The LLM-integrated app performing the target task, with no defense
target_app = PI.create_app(target_task, model, defense='no')

print("\n\n################ BASELINE (no attack) ################")
baseline_responses = []
for i, (data_prompt, ground_truth_label) in enumerate(target_app):
    response = target_app.query(data_prompt, verbose=1, idx=i, total=len(target_app))
    baseline_responses.append(response)

print("\n\n################ COMBINED ATTACK ################")
attack_responses = []
for i, (data_prompt, ground_truth_label) in enumerate(target_app):
    data_prompt_after_attack = attacker.inject(data_prompt, i, target_task=target_task.task)
    response = target_app.query(data_prompt_after_attack, verbose=1, idx=i, total=len(target_app))
    attack_responses.append(response)

evaluator = PI.create_evaluator(
    target_task_responses=baseline_responses,
    target_task=target_task,
    injected_task_responses=None,
    injected_task=attacker.task,
    attack_responses=attack_responses
)

print(f"\nPNA-T (clean-task accuracy, no attack) = {evaluator.pna_t}")
print(f"ASV   (attack success value)            = {evaluator.asv}")
print('[END]')
