
from learner.learn import Learn
from learner.constants import AdaptationLevel


try:
    model_learner = Learn()
    model_learner.get_true_model()
    model_learner.start_learning()
    if model_learner.ready.get_baseline() == AdaptationLevel.BASELINE_C:
        model_learner.dump_learned_model()

    model_learner.update_config_files()

    max_runs = 8
    if model_learner.ready.get_baseline() == AdaptationLevel.BASELINE_D:
        run=0
        while model_learner.budget > model_learner.used_budget:
            if run >= max_runs:
                break
            print("Online learning Run {}".format(run+1))
            model_learner.start_online_learning()
            model_learner.update_config_files()
            run+=1

    #print(model_learner.learner.measurePM(model_learner.default_conf))
except Exception as e:
    print("Error: {}".format(e))
