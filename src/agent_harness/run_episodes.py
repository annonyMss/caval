"""Online runs: AgentDojo or AgentDyn suites with the undefended agent, the DRIFT defense, or CAVAL's runtime verifier.
Adapted from the entry point of DRIFT (https://github.com/SaFo-Lab/DRIFT, MIT license, see LICENSE_DRIFT in this folder)."""
import time
from datetime import datetime
import argparse
import random
import numpy as np
import logging
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

from agent_harness.llm_client import OpenAIModel, OpenRouterModel, GoogleModel
from agent_harness.drift_baseline import *
from agent_harness.drift_baseline import DRIFTLLM, DRIFTTaskSuite, DRIFTToolsExecutionLoop
from caval.runtime_verifier import CavalVerifier

# ==== CLI args / seed / logger (was utils.py) ====
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    return seed

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            if getattr(handler, "_drift_file_handler", False):
                root_logger.removeHandler(handler)
                handler.close()
        handler = logging.FileHandler(filename)
        handler._drift_file_handler = True
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        root_logger.addHandler(handler)
    return logger

def get_args(description='DRIFT'):
    parser = argparse.ArgumentParser(description=description)
    # Eval Setting
    parser.add_argument('--benchmark_version', type=str, default='v1.2', help='the version of agentdojo')
    parser.add_argument('--model', type=str, default='gpt-4o-mini-2024-07-18', help='gpt-4o-mini, gpt-4o')
    parser.add_argument("--suites", type=str, default="banking,slack,travel,workspace", help="which suites to use, separated by comma.")
    parser.add_argument('--force_rerun', action='store_true', help='Whether to force rerun.')
    parser.add_argument('--run_id', type=str, default=None, help='Run identifier for separating logs and result files. Defaults to a timestamp.')
    parser.add_argument('--do_attack', action='store_true', help='Whether the setting is under attack.')
    parser.add_argument('--attack_type', type=str, default="important_instructions", help='The attack type, you can select from "direct, ignore_previous, system_message, injecagent, dos, swearwords_dos, captcha_dos, offensive_email_dos, felony_dos, important_instructions, important_instructions_no_user_name, important_instructions_no_model_name, important_instructions_no_names, important_instructions_wrong_model_name, important_instructions_wrong_user_name, tool_knowledge"')

    parser.add_argument('--target_user_tasks', type=str, default=None, help='User task number you want to evaluate, sperated by comma, such as "1,4,7".')
    parser.add_argument('--target_injection_tasks', type=str, default=None, help='Injection task number you want to specific evaluate, sperated by comma, such as "1,2,3".')

    # DRIFT Setting
    parser.add_argument("--build_constraints", action='store_true', help="Whether to build initial constraints.")
    parser.add_argument("--injection_isolation", action='store_true', help="Whether to detect injection instruction.")
    parser.add_argument("--dynamic_validation", action='store_true', help="Whether to validate dynamically.")
    parser.add_argument("--adaptive_attack", action='store_true', help="Whether to implement adaptive attack.")
    parser.add_argument("--caval", action='store_true',
                        help="Replace ToolsExecutor with CavalVerifier (constitution gates + calibrated "
                             "R-GCN score, allow/escalate/block per proposed call). Mutually meaningful "
                             "alongside plain undefended/DRIFT-defended runs for comparison.")
    parser.add_argument("--direct_attack", action='store_true',
                        help="E5 / adversarial-user mode: run each INJECTION task AS a user task "
                             "(no injected content anywhere). The attacker goal is what the user "
                             "asks for, so attacker success == task completion == the `utility` "
                             "field. NOTE `security` is meaningless here: run_task_with_pipeline "
                             "returns a hardcoded True when injection_task is None.")
    parser.add_argument("--checkpoint", type=str,
                        default="results/checkpoints/caval_scorer_seed1.pt",
                        help="CavalVerifier scorer checkpoint (the five deployed scorers "
                             "pass results/checkpoints/caval_scorer_seedN.pt).")
    parser.add_argument("--alpha_esc", type=float, default=0.30,
                        help="per-trace false-alarm budget of the escalation threshold (lam_esc from the checkpoint's cal_M)")
    parser.add_argument("--pairs_file", type=str, default=None,
                        help="JSON list of [user_task_id, injection_task_id] pairs; only these attacked episodes run (rewrite loop).")
    parser.add_argument("--rewrite_file", type=str, default=None,
                        help="JSON {\"user_task_X/injection_task_Y\": text}: replaces every injection string of that pair with text (rewrite loop).")
    parser.add_argument("--enforce", type=str, default="isolate", choices=["stop", "isolate"],
                        help="after an intervention: stop = end the episode; isolate = recovery (reject the call, remove the "
                             "tool returns that fed it, continue)")
    parser.add_argument("--reject_cap", type=int, default=3, help="interventions per episode under recovery before the episode ends")
    parser.add_argument("--no_gate", action="store_true",
                        help="the deterministic rules are logged but never decide (learned scorer only, the deployed configuration)")
    parser.add_argument("--no_score", action="store_true",
                        help="the learned score is logged but never decides (rules only, component analysis)")

    # Environment
    parser.add_argument('--seed', type=int, default=98, help='Random Seed.')


    args = parser.parse_args()

    return args
# ==== end utils ====


def main(args, suite_type):
    benchmark_version = args.benchmark_version
    suites = tuple(get_suites(benchmark_version).keys())
    suites = (suite_type,) # banking, slack, travel, workspace

    model_name = args.model

    # Set Attacker
    if args.do_attack:
        attacker = args.attack_type
    else:
        attacker = None

    model_output_name = f"{model_name}-adaptive_attack" if args.adaptive_attack else model_name
    if attacker is None:
        run_mode = "no_attack"
    else:
        run_mode = f"attack_{attacker}"

    save_dir = Path("data/runs") / model_output_name / run_mode / args.run_id / suites[0]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    logger_path = os.path.join(save_dir, "log.txt")
    logger = get_logger(logger_path)
    logger.info(f"Log File is saved at: {logger_path}")
    logger.info(f"Run Mode: {run_mode}")
    logger.info(f"Run ID: {args.run_id}")

    logger.info(f"Evaluating Suites: {suites}")

    if model_name.startswith("gpt-"):
        client = OpenAIModel(model=args.model, logger=logger)
        tools_pipeline_name = 'gpt-4o-2024-05-13'
        logger.info(f"Using OpenAI Client: {args.model}")

    elif model_name.startswith("gemini-"):
        client = GoogleModel(model=args.model, logger=logger)
        tools_pipeline_name = args.model
        logger.info(f"Using Google Client: {args.model}")

    else:
        client = OpenRouterModel(model=args.model, logger=logger)
        tools_pipeline_name = args.model
        logger.info(f"Using OpenRouter Client: {args.model}")
        # raise ValueError("Invalid model name.")

    llm = DRIFTLLM(args, client, logger=logger)

    caval_verifier = CavalVerifier(checkpoint_path=args.checkpoint,
                                       alpha_esc=args.alpha_esc,
                                       use_gate=not args.no_gate, use_score=not args.no_score,
                                       enforce=args.enforce, reject_cap=args.reject_cap) if args.caval else None
    executor_element = caval_verifier if args.caval else ToolsExecutor()
    tools_loop = DRIFTToolsExecutionLoop(
        [
            executor_element,
            llm,
        ]
    )
    tools_pipeline = AgentPipeline(
        [
            # SystemMessage("You are a helpful agent assistant with superior ."),
            InitQuery(),
            llm,
            tools_loop,
        ]
    )


    for suite_name in suites:
        suite = get_suite(benchmark_version, suite_name)
        task_suite = DRIFTTaskSuite(
            args,
            suite.name,
            suite.environment_type,
            suite.tools,
            suite.data_path,
            suite.benchmark_version,
            parent_instance = suite,
        )

    if args.target_user_tasks is None:
        tasks_to_run = task_suite.user_tasks.values()
        logger.info("Evaluate on all User Tasks.")

    else:
        target_user_task_id = args.target_user_tasks
        tasks_to_run = [task_suite.user_tasks[f"user_task_{task_id}"] for task_id in args.target_user_tasks.split(",")]
        logger.info(f"Evaluate on User Tasks of {target_user_task_id}.")


    if args.direct_attack:
        tasks_to_run = list(task_suite.injection_tasks.values())
        logger.info(f"DIRECT-ATTACK mode: {len(tasks_to_run)} injection tasks run as user tasks.")

    utility_result = []
    security_result = []
    tools_pipeline.name = tools_pipeline_name # ['meta-llama/Llama-3-70b-chat-hf', 'gemini-1.5-pro-002', 'claude-3-sonnet-20240229', command-r', 'command-r'-plus, 'gpt-3.5-turbo-0125', 'gpt-4o-2024-05-13', 'mistralai/Mixtral-8x7B-Instruct-v0.1']
    # tools_pipeline.name = "meta-llama/Llama-3-70b-chat-hf"

    resume_utility = 0
    resume_security = 0
    resume_total = 0
    if attacker is not None:
        logger.info(f"Using Attack Method: {attacker}")
        attack = load_attack(attacker, task_suite, tools_pipeline)
        target_injection_tasks = args.target_injection_tasks
        if target_injection_tasks is not None:
            injection_tasks_to_run = {
            injection_task_id: suite.get_injection_task_by_id(injection_task_id)
            for injection_task_id in args.target_injection_tasks.split(",")
            }
            logger.info(f"Injection Tasks of {target_injection_tasks}.")
        else:
            logger.info("Evaluate on all injection tasks.")
            injection_tasks_to_run = task_suite.injection_tasks

        pairs = {tuple(p) for p in json.load(open(args.pairs_file))} if args.pairs_file else None
        rewrites = json.load(open(args.rewrite_file)) if args.rewrite_file else {}
        for idx, user_task in enumerate(tasks_to_run):
            user_task_name = user_task.ID
            match = re.fullmatch(r'user_task_(\d+)', user_task_name)
            user_task_idx = int(match.group(1))
            for injec_idx, injection_task_id in enumerate(injection_tasks_to_run):
                match = re.fullmatch(r'injection_task_(\d+)', injection_task_id)
                injection_task_idx = int(match.group(1))
                if pairs is not None and (user_task_name, injection_task_id) not in pairs:
                    continue
                pre_total_tokens = llm.client.total_tokens

                result_file_path = Path(save_dir) / f"user_task_{user_task_idx}" / attacker / f"injection_task_{injection_task_idx}.json"
                result_file_path.parent.mkdir(parents=True, exist_ok=True)
                if not args.force_rerun and os.path.exists(result_file_path):
                    try:
                        with open(result_file_path, "r", encoding="utf-8") as f:
                            loaded_result = json.load(f)
                            if "utility" in loaded_result and "security" in loaded_result:
                                utility_result.append(loaded_result["utility"])
                                security_result.append(loaded_result["security"])
                                logger.info(f"user_task_{user_task_idx} with injection_task_{injection_task_idx} result already exists; skipping run.\nAttack Success Ratio: {security_result.count(True) + resume_security} / {len(security_result) + resume_total}\nUtility Success Ratio: {utility_result.count(True) + resume_utility} / {len(utility_result) + resume_total}")
                                continue

                    except Exception as e:
                        logger.info(f"Loading existing result file at {result_file_path} failed: {e}. The task will be re-run.")
                        logger.info(f"Re-runing user_task_{user_task_idx}-injection_task_{injection_task_idx} ...")


                injection_task = suite.get_injection_task_by_id(injection_task_id)
                task_injections = attack.attack(user_task, injection_task)
                rewritten = rewrites.get(f"{user_task_name}/{injection_task_id}")

              
                if rewritten is not None:
                    task_injections = {key: rewritten for key in task_injections}

                if caval_verifier is not None:
                    caval_verifier.reset()
                start_time = time.time()
                utility, security, messages = task_suite.run_task_with_pipeline(tools_pipeline, user_task, injection_task, task_injections)
                end_time = time.time()
                utility_result.append(utility)
                security_result.append(security)
                with open(result_file_path, "w") as f:
                    json.dump({"suite_name": suite_type, "pipeline_name": f"{args.model}", "run_mode": run_mode, "run_id": args.run_id, "user_task_id": f"user_task_{user_task_idx}", "injection_task_id": f"injection_task_{injection_task_idx}", "attack_type": f"{attacker}", "build_constraints": args.build_constraints, "injection_isolation": args.injection_isolation, "dynamic_validation": args.dynamic_validation, "adaptive_attack": args.adaptive_attack, "caval": args.caval, "enforce": args.enforce if args.caval else None, "reject_cap": args.reject_cap if args.caval else None, "rewritten_injection": rewritten, "verifier_log": caval_verifier.log if caval_verifier is not None else None, "tool_permission": llm.tool_permissions, "initial_trajectory": llm.initial_function_trajectory, "initial_checklist": llm.initial_node_checklist, "conversations": messages, "benchmark_version": args.benchmark_version, "temperature": args.temperature, "utility": utility, "security": security, "total_tokens": llm.client.total_tokens - pre_total_tokens, "duration": end_time - start_time}, f, indent=4)

                logger.info(f"user_task_{user_task_idx} with injection_task_{injection_task_idx} Utility Success Ratio: {utility_result.count(True) + resume_utility} / {len(utility_result) + resume_total}")
                logger.info(f"user_task_{user_task_idx} with injection_task_{injection_task_idx} Attack Success Ratio: {security_result.count(True) + resume_security} / {len(security_result) + resume_total}")

 
    else:
        logger.info("Evaluating on User Tasks.")
        for idx, user_task in enumerate(tasks_to_run):
            user_task_name = user_task.ID
            # accept injection_task_N too: --direct_attack feeds injection tasks
            # through this same benign loop (their ID keeps the injection_task_
            # prefix, which the old user_task-only regex could not match).
            match = re.fullmatch(r'(user_task|injection_task)_(\d+)', user_task_name)
            task_label = f"{match.group(1)}_{int(match.group(2))}"
            user_task_idx = int(match.group(2))
            pre_total_tokens = llm.client.total_tokens

            result_file_path = Path(save_dir) / task_label / "none" / f"none.json"
            result_file_path.parent.mkdir(parents=True, exist_ok=True)
            if not args.force_rerun and os.path.exists(result_file_path):
                try:
                    with open(result_file_path, "r", encoding="utf-8") as f:
                        loaded_result = json.load(f)
                        if "utility" in loaded_result and "security" in loaded_result:
                            utility_result.append(loaded_result["utility"])
                            security_result.append(loaded_result["security"])
                            logger.info(f"user_task_{user_task_idx} result already exists; skipping run.\nAttack Success Ratio: {security_result.count(True) + resume_security} / {len(security_result) + resume_total}\nUtility Success Ratio: {utility_result.count(True) + resume_utility} / {len(utility_result) + resume_total}")
                            continue

                except Exception as e:
                    logger.info(f"Loading existing result file at {result_file_path} failed: {e}. The task will be re-run.")
                    logger.info(f"Re-runing user_task_{user_task_idx} ...")


            if caval_verifier is not None:
                caval_verifier.reset()
            start_time = time.time()
            utility, security, messages = task_suite.run_task_with_pipeline(tools_pipeline, user_task, injection_task=None, injections={})
            end_time = time.time()
            utility_result.append(utility)
            security_result.append(security)
            with open(result_file_path, "w") as f:
                json.dump({"suite_name": suite_type, "pipeline_name": f"{args.model}", "run_mode": run_mode, "run_id": args.run_id, "user_task_id": task_label, "injection_task_id": None, "attack_type": None, "direct_attack": args.direct_attack, "build_constraints": args.build_constraints, "injection_isolation": args.injection_isolation, "dynamic_validation": args.dynamic_validation, "adaptive_attack": args.adaptive_attack, "caval": args.caval, "enforce": args.enforce if args.caval else None, "reject_cap": args.reject_cap if args.caval else None, "verifier_log": caval_verifier.log if caval_verifier is not None else None, "tool_permission": llm.tool_permissions, "initial_trajectory": llm.initial_function_trajectory, "initial_checklist": llm.initial_node_checklist, "conversations": messages, "benchmark_version": args.benchmark_version, "temperature": args.temperature, "utility": utility, "security": security, "total_tokens": llm.client.total_tokens - pre_total_tokens, "duration": end_time - start_time}, f, indent=4)

                logger.info(f"user_task_{user_task_idx} Utility Success Ratio: {utility_result.count(True) + resume_utility} / {len(utility_result) + resume_total}")

    logger.info(f"Overall Utility Success Ratio: {(utility_result.count(True) + resume_utility) / (len(utility_result) + resume_total)}")
    logger.info(f"Overall Attack Success Ratio: {(security_result.count(True) + resume_security) / (len(security_result) + resume_total)}")
    logger.info(f"{suite_type} Done!")

if __name__ == "__main__":
    args = get_args()
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_seed(args.seed)
    suites = args.suites.split(",")
    for suite_type in suites:
        main(args, suite_type)