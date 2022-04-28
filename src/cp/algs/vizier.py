'''
Created on Apr 27, 2022

@author: immanueltrummer
'''
import cp.text.fact
import datetime
import googleapiclient.discovery
import google.cloud.storage
import os
import time


class VizierSum():
    """ Baseline using Google's Vizier to generate summaries. """
    
    def __init__(self, nr_facts, nr_preds, pred_cnt, agg_cnt):
        """ Initializes summary generation for specific dimensions.
        
        Args:
            nr_facts: number of facts in summaries
            nr_preds: maximal number of predicates per fact
            pred_cnt: number of predicate options
            agg_cnt: number of aggregate options
        """
        self.nr_facts = nr_facts
        self.nr_preds = nr_preds
        self.pred_cnt = pred_cnt
        self.agg_cnt = agg_cnt
        
        self.project_id = os.environ['VIZIER_PROJECT_ID']
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.study_id = f'itrummer_study_{timestamp}'
        self.study_parent = f'projects/{self.project_id}/locations/us-central1'
        self.study_name = f'{self.study_parent}/studies/{self.study_id}'
        self.client_id = 'summary-client'

        self.client = self._client()
        self.operations = self.client.projects().locations().operations()
        self.studies = self.client.projects().locations().studies()
        self.trials = self.studies.trials()
        study_config = self._study_config(nr_facts, nr_preds, pred_cnt, agg_cnt)
        self._create_study(study_config)
    
    def register_feedback(self, trial_name, quality):
        """ Provide feedback on suggested parameter values. 
        
        Args:
            trial_name: name of evaluated trial
            quality: quality of evaluated summary
        """
        metric = {'metric': 'quality', 'value': quality}
        measurement = {'step_count': 1, 'metrics': [metric,]}
        self.trials.addMeasurement(
            name=trial_name, 
            body={'measurement': measurement}).execute()
        self.trials.complete(name=trial_name).execute()

    def suggest_summary(self):
        """ Generate suggestions for summaries to evaluate. 
        
        Returns:
            trial name and associated summary
        """
        body = {'client_id': self.client_id, 'suggestion_count': 1}
        trial_response = self.trials.suggest(
            parent=self.study_name, body=body).execute()
        op_id = trial_response['name'].split('/')[-1]
        op_name = f'{self.study_parent}/operations/{op_id}'
        
        get_op = self.operations.get(name=op_name)
        while not get_op.execute().get('done', False):
            time.sleep(1)
        
        trial_info = get_op.execute()['response']['trials'][0]
        trial_id = int(trial_info['name'].split('/')[-1])
        trial_name = f'{self.study_parent}/studies/{self.study_id}/trials/{trial_id}'
        trial = self.trials.get(name=trial_name).execute()
        
        return trial_name, self._extract_summary(trial)
    
    def _client(self):
        """ Generates client to access Vizier. 
        
        Returns:
            client for accessing ML API
        """
        client = google.cloud.storage.Client(self.project_id)
        bucket = client.get_bucket('caip-optimizer-public')
        blob = bucket.get_blob('api/ml_public_google_rest_v1.json')
        api_doc = blob.download_as_string()
        return googleapiclient.discovery.build_from_document(api_doc)

    def _create_study(self, study_config):
        """ Generates a new study. 
        
        Args:
            study_config: configuration of study to create
        """
        request = self.studies.create(
            parent=self.study_parent, 
            studyId=self.study_id, 
            body=study_config)
        request.execute()
    
    def _extract_summary(self, trial):
        """ Extract summary from trial. 
        
        Args:
            trial: suggested parameter values, describing summary
        
        Returns:
            summary specified by trial suggestions
        """
        summary = []
        for _ in range(self.nr_preds):
            fact = cp.text.fact.Fact(self.nr_preds)
            summary.append(fact)
        
        for p in trial['parameters']:
            name = p['parameter']
            properties = name.split('-')
            fact_idx = int(properties[1])
            fact = summary[fact_idx]
            
            value = int(p['intValue'])
            p_type = properties[0]
            if p_type == 'A':
                fact.set_agg(value)
            elif p_type == 'P':
                pred_idx = int(properties[2])
                fact.set_pred(pred_idx, value)
            else:
                raise ValueError(f'Unsupported parameter type: {p_type}')
        
        return summary
    
    def _study_config(self, nr_facts, nr_preds, pred_cnt, agg_cnt):
        """ Generates configuration for study.
        
        Args:
            nr_facts: number of facts in summary
            nr_preds: number of possible predicates
            pred_cnt: number of predicate options
            agg_cnt: number of aggregate options
        
        Returns:
            a study configuration (as Python dictionary)
        """
        study = {'algorithm': 'ALGORITHM_UNSPECIFIED'}
        parameters = []
        study['parameters'] = parameters
        for fact_idx in range(nr_facts):
            
            p = {}
            p['parameter'] = f'A-{fact_idx}'
            p['type'] = 'INTEGER'
            int_value_spec = {}
            int_value_spec['min_value'] = 0
            int_value_spec['max_value'] = agg_cnt - 1
            p['integer_value_spec'] = int_value_spec
            parameters.append(p)
            
            for pred_idx in range(nr_preds):
                p = {}
                p['parameter'] = f'P-{fact_idx}-{pred_idx}'
                p['type'] = 'INTEGER'
                int_value_spec = {}
                int_value_spec['min_value'] = 0
                int_value_spec['max_value'] = pred_cnt - 1
                p['integer_value_spec'] = int_value_spec
                parameters.append(p)

        metric = {}
        study['metrics'] = [metric]
        metric['metric'] = 'quality'
        metric['goal'] = 'MAXIMIZE'
        
        return {'study_config': study}