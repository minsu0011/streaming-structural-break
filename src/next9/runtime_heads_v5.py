"""Frozen heads with exact empty-rule design compilation; same model numerics."""
from dataclasses import dataclass
import numpy as np
from src.next9.fast_design import frozen_design
from src.next9.streaming import classifier_probabilities
from src.next9.cross import path_features as cross_path

@dataclass
class FrozenHeads:
    spec: dict
    models: dict
    self_laws: dict
    cross_laws: dict
    m6: object
    tabular: object
    correction: object

    def predict(self,bank,baseline=None,include_tabular=True):
        s=self.spec;rows=np.asarray([bank.time],dtype=np.int64)
        rules={int(k):v for k,v in s['rule_specs'].items()}
        run_maps={int(k):np.asarray(v,dtype=int) for k,v in s['run_maps'].items()}
        x,names=frozen_design(self,bank)
        if tuple(s['input_names'])!=names:raise ValueError('Frozen online M input schema changed')
        values={method:float(np.float32(classifier_probabilities(model,x[:,s['selected_columns'][method]])[0,1]))
                for method,model in self.models.items()}
        result={method:values[alias] for method,alias in s['aliases'].items()}
        parts=[x]
        for j,law in self.cross_laws.items():parts.append(cross_path(bank,rows,law))
        cx=np.column_stack(parts).astype(np.float32)
        result['M6']=result['M5'] if self.m6 is None else float(np.float32(classifier_probabilities(self.m6,cx)[0,1]))
        if include_tabular:result['TABULAR_SAME_INPUT']=float(np.float32(classifier_probabilities(self.tabular,cx)[0,1]))
        if baseline is not None:
            result['M0']=float(baseline)
            result['M7']=float(baseline) if self.correction is None else float(self.correction.predict(np.asarray([baseline]),np.asarray([result['M5']],dtype=np.float32))[0])
        return result

    def predict_M7(self,bank,baseline):
        if self.correction is None:return baseline
        s=self.spec;rows=np.asarray([bank.time],dtype=np.int64)
        rules={int(k):v for k,v in s['rule_specs'].items()}
        mapping={int(k):np.asarray(v,dtype=int) for k,v in s['run_maps'].items()}
        x,names=frozen_design(self,bank)
        if tuple(s['input_names'])!=names:raise ValueError('Frozen online M input schema changed')
        method=s['aliases']['M5'];model=self.models[method]
        p=np.float32(classifier_probabilities(model,x[:,s['selected_columns'][method]])[0,1])
        return float(self.correction.predict(np.asarray([baseline]),np.asarray([p]))[0])

