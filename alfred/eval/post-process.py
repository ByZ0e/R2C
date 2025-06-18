import os
import glob
import json

def read_data():
    result_list = []
    success_list = []
    failure_list = []
    num_success = 0
    num_failure = 0
    print('Indexing result in {}'.format(result_dir))
    for partition in partitions:
        for dir_name in sorted(
                glob.glob(os.path.join(result_dir, partition, '*/*'))):
            if 'trial_' in os.path.basename(dir_name):
                result_path = os.path.join(dir_name, 'log.txt')
                if not os.path.isfile(result_path):
                    continue
                traj = '/'.join(result_path.split('/')[-4:-1])
                result_list.append(traj)
                
                with open(result_path, 'r') as f:
                    try:
                        result = eval(f.readlines()[-1])
                        success = result['success']
                        if success:
                            success_list.append(traj)
                            num_success += 1
                        else:
                            failure_list.append(traj)
                            num_failure += 1
                    except:
                        result_list.remove(traj)
    num_files = len(result_list)
    print('# eval trajs: ', num_files)
    print('SR: {}/{}={}'.format(num_success, num_files, num_success/num_files))
    
    with open(os.path.join(result_dir, 'success_list.json'), 'w') as f:
        json.dump(success_list,f)
    with open(os.path.join(result_dir, 'failure_list.json'), 'w') as f:
        json.dump(failure_list,f)
        
# result_dir = 'eval_subgoal_results_0504/eval_subgoal_v2_0-250_2024-05-03-23:57:28'
# result_dir = 'eval_subgoal_results_0504/eval_subgoal_v2_250-506_2024-05-04-00:00:12'
result_dir = 'eval_subgoal_results_0504/eval_subgoal_v1_250-506_2024-05-04-01:03:08'
partitions = ['valid_seen', 'valid_unseen']
read_data()