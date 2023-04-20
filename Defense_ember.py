from mw_backdoor.defense_tool import Defense_3
import argparse
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--target', type=str, default='combined')
args = parser.parse_args()
EMBER = 600000
### strategy should be combined, independent or no_attack
strategy = args.target
if strategy == "pecan":
    defense = Defense_3(0.001, None,
                        "/nobackup/yuhao_data/malware_poison/embernn_fig2_3_0.001_nd_na/ember__embernn__combined_shap__combined_shap__all/watermarked_train",
                        None,
                        "/nobackup/yuhao_data/malware_poison/embernn_fig2_3_0.001_nd_na/ember__embernn__combined_shap__combined_shap__all/watermarked_X_test.npy",
                        strategy,
                        "/nobackup/yuhao_data/malware_poison/embernn_fig2_3_0.001_nd_na/ember__embernn__combined_shap__combined_shap__all/wm_config.npy",
                        dataset='ember', surrogate_model='DNN', wm_size=3)
    defense.run_defense()
    print(defense.show())

elif strategy != 'no_attack':
    wm_size = [8, 17]
    poison_rate = [0.01, 0.02, 0.04]
    for size in wm_size:
        for p_r in poison_rate:
            backdoor_model_pos = f'./backdoor_ember_{size}/{strategy}/{int(EMBER * p_r)}_ember_lightgbm_backdoored'
            X_train_pos = f'./backdoor_ember_{size}/{strategy}/{int(EMBER * p_r)}_watermarked_X.npy'
            y_train_pos = f'./backdoor_ember_{size}/{strategy}/{int(EMBER * p_r)}_watermarked_y.npy'
            X_test_pos = f'./backdoor_ember_{size}/{strategy}/{int(EMBER * p_r)}_watermarked_X_test.npy'
            config_pos = f'./backdoor_ember_{size}/{strategy}/{int(EMBER * p_r)}_wm_config.npy'
            defense = Defense_3(p_r, backdoor_model_pos, X_train_pos, y_train_pos, X_test_pos, strategy, config_pos,
                                dataset='ember', surrogate_model='lightgbm', wm_size=size)
            defense.run_defense()
            print(defense.show())

else:
    wm_size = 0
    p_r = 0
    backdoor_model_pos = None
    X_train_pos = f'./dataset/x_train.npy'
    y_train_pos = f'./dataset/y_train.npy'
    X_test_pos = f'./dataset/x_test.npy'
    config_pos = None
    defense = Defense_3(p_r, backdoor_model_pos, X_train_pos, y_train_pos, X_test_pos, strategy, config_pos,
                        dataset='ember', surrogate_model='lightgbm', wm_size=wm_size)
    defense.run_defense()
    print(defense.show())
