"""Empty previous-issued batches have no observations, not an estimator error."""
from dataclasses import dataclass
import numpy as np
from src.next9.streaming import classifier_probabilities
from src.next9.transitions import design as self_design
from src.next9.cross import design as cross_design


@dataclass
class RuntimeSelfLaw:
    law: object

    @property
    def table(self):return self.law.table

    def probabilities(self,state,run,slope,context):
        state=np.asarray(state,dtype=int);x=self_design(state,run,slope,context)
        if not len(state):return np.empty((0,4),dtype=float)
        if self.law.classifier is None:return self.law.table[state]
        return classifier_probabilities(self.law.classifier,x)


@dataclass
class RuntimeCrossLaw:
    law: object

    @property
    def prior(self):return self.law.prior

    @property
    def destination(self):return self.law.destination

    @property
    def parents(self):return self.law.parents

    def probabilities(self,bank,rows):
        if not len(rows):return np.empty((0,4),dtype=float)
        return classifier_probabilities(self.law.classifier,cross_design(bank,rows,self.destination,self.parents))
