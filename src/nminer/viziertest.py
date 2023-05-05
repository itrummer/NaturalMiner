'''
Created on Apr 25, 2022

@author: immanueltrummer
'''
import json
import time
import datetime
from googleapiclient import errors

from google.cloud import storage
from googleapiclient import discovery

PROJECT_ID = 'baboons-348402'
_OPTIMIZER_API_DOCUMENT_BUCKET = 'caip-optimizer-public'
_OPTIMIZER_API_DOCUMENT_FILE = 'api/ml_public_google_rest_v1.json'

def read_api_document():
    client = storage.Client(PROJECT_ID)
    bucket = client.get_bucket(_OPTIMIZER_API_DOCUMENT_BUCKET)
    blob = bucket.get_blob(_OPTIMIZER_API_DOCUMENT_FILE)
    return blob.download_as_string()

ml = discovery.build_from_document(service=read_api_document())
print('Successfully built the client.')


# Update to your username
USER = 'itrummer' #@param {type: 'string'}

# These will be automatically filled in.
STUDY_ID = '{}_study_{}'.format(
    USER, datetime.datetime.now().strftime(
        '%Y%m%d_%H%M%S')) #@param {type: 'string'}
REGION = 'us-central1'

def study_parent():
    return 'projects/{}/locations/{}'.format(
        PROJECT_ID, REGION)

def study_name(study_id):
    return 'projects/{}/locations/{}/studies/{}'.format(
        PROJECT_ID, REGION, study_id)

def trial_parent(study_id):
    return study_name(study_id)

def trial_name(study_id, trial_id):
    return 'projects/{}/locations/{}/studies/{}/trials/{}'.format(
        PROJECT_ID, REGION, study_id, trial_id)

def operation_name(operation_id):
    return 'projects/{}/locations/{}/operations/{}'.format(
        PROJECT_ID, REGION, operation_id)


print('USER: {}'.format(USER))
print('PROJECT_ID: {}'.format(PROJECT_ID))
print('REGION: {}'.format(REGION))
print('STUDY_ID: {}'.format(STUDY_ID))


# Parameter Configuration
param_r = {
    'parameter': 'r',
    'type' : 'DOUBLE',
    'double_value_spec' : {
        'min_value' : 0,
        'max_value' : 1
    }
}

param_theta = {
    'parameter': 'theta',
    'type' : 'DOUBLE',
    'double_value_spec' : {
        'min_value' : 0,
        'max_value' : 1.57
    }
}

# Objective Metrics
metric_y1 = {
    'metric' : 'y1',
    'goal' : 'MINIMIZE'
}

metric_y2 = {
    'metric' : 'y2',
    'goal' : 'MAXIMIZE'
}

# Put it all together in a study configuration
study_config = {
    'algorithm' : 'ALGORITHM_UNSPECIFIED',  # Let the service choose the `default` algorithm.
    'parameters' : [param_r, param_theta,],
    'metrics' : [metric_y1, metric_y2,],
}

study = {'study_config': study_config}
print(json.dumps(study, indent=2, sort_keys=True))


# Creates a study
req = ml.projects().locations().studies().create(
    parent=study_parent(), studyId=STUDY_ID, body=study)
try :
    print(req.execute())
except errors.HttpError as e:
    if e.resp.status == 409:
        print('Study already existed.')
    else:
        raise e


import math


# r * sin(theta)
def Metric1Evaluation(r, theta):
    """Evaluate the first metric on the trial."""
    return r * math.sin(theta)


# r * cose(theta)
def Metric2Evaluation(r, theta):
    """Evaluate the second metric on the trial."""
    return r * math.cos(theta)


def CreateMeasurement(trial_id, r, theta):
    print(("=========== Start Trial: [{0}] =============").format(trial_id))
    
    # Evaluate both objective metrics for this trial
    y1 = Metric1Evaluation(r, theta)
    y2 = Metric2Evaluation(r, theta)
    print('[r = {0}, theta = {1}] => y1 = r*sin(theta) = {2}, y2 = r*cos(theta) = {3}'.format(r, theta, y1, y2))
    metric1 = {'metric': 'y1', 'value': y1}
    metric2 = {'metric': 'y2', 'value': y2}
    
    # Return the results for this trial
    measurement = {'step_count': 1, 'metrics': [metric1, metric2,]}
    return measurement


client_id = 'client1' #@param {type: 'string'}
suggestion_count_per_request =  5 #@param {type: 'integer'}
max_trial_id_to_stop =  50 #@param {type: 'integer'}

print('client_id: {}'.format(client_id))
print('suggestion_count_per_request: {}'.format(suggestion_count_per_request))
print('max_trial_id_to_stop: {}'.format(max_trial_id_to_stop))


trial_id = 0
while trial_id < max_trial_id_to_stop:
    # Requests trials.
    resp = ml.projects().locations().studies().trials().suggest(
      parent=trial_parent(STUDY_ID),
      body={'client_id': client_id, 'suggestion_count': suggestion_count_per_request}).execute()
    op_id = resp['name'].split('/')[-1]
    
    # Polls the suggestion long-running operations.
    get_op = ml.projects().locations().operations().get(name=operation_name(op_id))
    while True:
        operation = get_op.execute()
        if 'done' in operation and operation['done']:
            break
        time.sleep(1)
    
    for suggested_trial in get_op.execute()['response']['trials']:
        trial_id = int(suggested_trial['name'].split('/')[-1])
        # Featches the suggested trials.
        locations = ml.projects().locations()
        trial = locations.studies().trials().get(name=trial_name(STUDY_ID, trial_id)).execute()
        if trial['state'] in ['COMPLETED', 'INFEASIBLE']:
            continue
        
    params = {}
    for param in trial['parameters']:
        if param['parameter'] == 'r':
            r = param['floatValue']
        elif param['parameter'] == 'theta':
            theta = param['floatValue']
    
    # Evaluates trials and reports measurement.
    ml.projects().locations().studies().trials().addMeasurement(
        name=trial_name(STUDY_ID, trial_id),
        body={'measurement': CreateMeasurement(trial_id, r, theta)}).execute()
    # Completes the trial.
    ml.projects().locations().studies().trials().complete(
        name=trial_name(STUDY_ID, trial_id)).execute()



max_trials_to_annotate = 20

import matplotlib.pyplot as plt
trial_ids = []
y1 = []
y2 = []
resp = ml.projects().locations().studies().trials().list(parent=trial_parent(STUDY_ID)).execute()
for trial in resp['trials']:
    if 'finalMeasurement' in trial:
        trial_ids.append(int(trial['name'].split('/')[-1]))
        metrics = trial['finalMeasurement']['metrics']
        try:
            y1.append([m for m in metrics if m['metric'] == "y1"][0]['value'])
            y2.append([m for m in metrics if m['metric'] == "y2"][0]['value'])
        except:
            pass

fig, ax = plt.subplots()
ax.scatter(y1, y2)
plt.xlabel("y1=r*sin(theta)")
plt.ylabel("y2=r*cos(theta)");
for i, trial_id in enumerate(trial_ids):
    # Only annotates the last `max_trials_to_annotate` trials
    if i > len(trial_ids) - max_trials_to_annotate:
        try:
            ax.annotate(trial_id, (y1[i], y2[i]))
        except:
            pass
plt.gcf().set_size_inches((16, 16))